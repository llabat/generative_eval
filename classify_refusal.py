"""
Run the encoder-based refusal classifier over a CSV of generations and add
`refusal_pred` / `non_normal` columns.

This is a standalone version of the "Refusal Classifier" section of
generative_eval.ipynb. Takes a config with a `refusal_classifier:` section,
in either of two shapes:

  - Embedded in a full pipeline config (as produced by generate_texts.py /
    run_pipeline.py -- see pipeline_config.py's docstring), identified by
    the presence of `output_dir:` at the top level:

        output_dir: runs/some_model
        refusal_classifier:
          checkpoint: /path/to/refusal/checkpoint
          batch_size: 256         # optional, default 256
          max_length: 4096        # optional, default 4096

    Reads <output_dir>/final.csv if it already exists (e.g.
    classify_stance.py ran first), else <output_dir>/generations.csv, and
    writes/overwrites <output_dir>/final.csv -- so classify_stance.py and
    classify_refusal.py can run in either order, or just one of them,
    without clobbering each other's column.

  - Standalone, just the refusal_classifier section, no output_dir --
    point it directly at any CSV with `prompt`/`response` columns:

        refusal_classifier:
          checkpoint: /path/to/refusal/checkpoint
          input_csv: /path/to/generations.csv
          output_csv: /path/to/generations_classified.csv   # optional,
                                                              #  defaults to
                                                              #  <input_csv
                                                              #  minus .csv>
                                                              #  _classified.csv
          batch_size: 256
          max_length: 4096

This encoder classifier is far lighter than the generation/stance models,
so it's meant to be run as its own (small-GPU) job -- see run_pipeline.py.

Usage:
    python classify_refusal.py config.yml
"""

import os
import sys

import pandas as pd
import torch
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append("/home/labat/refusal")
from architecture import RefusalClassifier  # noqa: E402

TOKENIZE_CHUNK_SIZE = 4000


def load_refusal_config(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if "refusal_classifier" not in cfg:
        raise ValueError(f"{config_path} has no refusal_classifier section")
    return cfg


def resolve_io_paths(cfg, config_path):
    """Returns (input_csv, output_csv), picking the pipeline-embedded
    convention (final.csv/generations.csv under output_dir) if `output_dir`
    is present, else the standalone input_csv/output_csv under
    refusal_classifier.
    """
    rc_cfg = cfg["refusal_classifier"]

    if "output_dir" in cfg:
        final_csv = os.path.join(cfg["output_dir"], "final.csv")
        generations_csv = os.path.join(cfg["output_dir"], "generations.csv")
        input_csv = final_csv if os.path.exists(final_csv) else generations_csv
        return input_csv, final_csv

    if "input_csv" not in rc_cfg:
        raise ValueError(
            f"{config_path}: refusal_classifier.input_csv is required when the "
            "config has no top-level output_dir (standalone mode)"
        )
    input_csv = rc_cfg["input_csv"]
    default_output_csv = (
        input_csv[:-len(".csv")] + "_classified.csv" if input_csv.endswith(".csv")
        else input_csv + "_classified.csv"
    )
    output_csv = rc_cfg.get("output_csv", default_output_csv)
    return input_csv, output_csv


def build_text_column(df):
    return "PROMPT:\n" + df["prompt"].fillna("") + "\n\nRESPONSE:\n" + df["response"].fillna("")


def classify_refusal(df, ckpt, device, max_length, batch_size):
    """Adds/overwrites `refusal_pred` (and returns df) by running the
    RefusalClassifier over df["text"], batching same-length examples
    together for speed via a length-sorted order.
    """
    texts = df["text"].tolist()
    missing_input = (df["prompt"].isna() | df["response"].isna()).tolist()

    print(f"Loading refusal classifier from {ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = RefusalClassifier.from_pretrained(ckpt).to(device)
    model.eval()
    idx2label = model.config.id2label
    label_map = {v: k for k, v in idx2label.items()}
    print(f"Checkpoint labels: {sorted(label_map)}")

    input_ids, attention_mask = [], []
    for i in tqdm(range(0, len(texts), TOKENIZE_CHUNK_SIZE), desc="tokenizing"):
        chunk_enc = tokenizer(
            texts[i:i + TOKENIZE_CHUNK_SIZE], truncation=True, max_length=max_length
        )
        input_ids.extend(chunk_enc["input_ids"])
        attention_mask.extend(chunk_enc["attention_mask"])

    lengths = [len(ids) for ids in input_ids]
    order = sorted(range(len(texts)), key=lambda i: lengths[i])

    row_ids = df.index.tolist()
    truncated = [l >= max_length for l in lengths]
    truncated_row_ids = [row_ids[i] for i, t in enumerate(truncated) if t]
    print(f"{len(truncated_row_ids)}/{len(texts)} rows truncated at {max_length} tokens")

    missing_input_row_ids = [row_ids[i] for i, m in enumerate(missing_input) if m]
    print(f"{len(missing_input_row_ids)}/{len(texts)} rows had a missing prompt or response")

    preds = [None] * len(texts)
    with torch.no_grad():
        for i in tqdm(range(0, len(order), batch_size), desc="predicting"):
            batch_idx = order[i:i + batch_size]
            batch_features = [
                {"input_ids": input_ids[j], "attention_mask": attention_mask[j]}
                for j in batch_idx
            ]

            # Pads already-tokenized sequences to the batch's own max length
            # -- no re-tokenization of text involved.
            enc = tokenizer.pad(batch_features, padding=True, return_tensors="pt").to(device)

            logits = model(**enc).logits
            batch_preds = torch.argmax(logits, dim=1).cpu().tolist()

            for j, p in zip(batch_idx, batch_preds):
                preds[j] = p

    df["refusal_pred"] = preds
    df["refusal_pred"] = df["refusal_pred"].map(idx2label)
    df["non_normal"] = df["refusal_pred"] != "NORMAL"
    return df


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: python {sys.argv[0]} config.yml")
    config_path = sys.argv[1]
    cfg = load_refusal_config(config_path)
    rc_cfg = cfg["refusal_classifier"]

    input_csv, output_csv = resolve_io_paths(cfg, config_path)

    print(f"Loading {input_csv}")
    df = pd.read_csv(input_csv)
    for col in ("prompt", "response"):
        if col not in df.columns:
            raise ValueError(f"Expected a '{col}' column in {input_csv}")

    df["text"] = build_text_column(df)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = classify_refusal(
        df,
        ckpt=rc_cfg["checkpoint"],
        device=device,
        max_length=rc_cfg.get("max_length", 4096),
        batch_size=rc_cfg.get("batch_size", 256),
    )

    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} rows with refusal predictions to {output_csv}")


if __name__ == "__main__":
    main()
