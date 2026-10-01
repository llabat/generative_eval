"""
One-off diagnostic, not part of the pipeline: re-run the templ-5 stance judge
on a stratified sample (by template_id) of an existing final.csv, keeping the
judge's RAW completion (not just classify_stance.parse_label()'s parsed
label) alongside the original stance_pred it produced during the real run.

Two things this is for:
  1. Gauge how often the judge's direction looks wrong on hand inspection
     (agreement rate against a human reading of stance vs. response), across
     template_ids, not just the handful of oppose examples spot-checked so
     far.
  2. Check whether errors are a *parsing* problem: with max_new_tokens=64 the
     judge may reason before answering, and parse_label()'s fallback takes
     the LAST standalone "1".."5"/"refusal" token in the completion -- if a
     completion gets cut off by the token budget mid-reasoning (before it
     commits to a final answer), that fallback could be grabbing an
     incidental digit mentioned while reasoning, not the judge's real
     verdict. Having the raw completion in front of us settles that per row
     (deterministic decoding -- do_sample=False -- means the original
     stance_pred should exactly reproduce here too; if it doesn't, something
     other than the judge's own non-determinism is at play).

Optional additions (defaults reproduce the original behaviour exactly):
  --judge-model / --template-id / --templates-csv  swap the judge and template
  --backend vllm --tensor-parallel-size N          judges too big for one GPU
                                                   (e.g. a 70B, N=2), via
                                                   vllm_backend.py
  --sample-size N                                  plain uniform random sample of N rows
                                                   instead of n_per_template per
                                                   template_id; streams the CSV in
                                                   chunks, so it also works on huge
                                                   final.csv files. If the input has
                                                   no stance_pred column (stance stage
                                                   never ran on it), the
                                                   reproduced_original comparison is
                                                   skipped.

Usage:
    python diagnose_stance_judge.py <input_final_csv> <output_csv> [n_per_template]
        [--judge-model PATH] [--template-id ID] [--templates-csv CSV]
        [--backend transformers|vllm] [--tensor-parallel-size N] [--sample-size N]
        [--chat-template-kwargs '{"enable_thinking": false}']
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classify_stance import load_template, parse_label

JUDGE_MODEL = "allenai/Llama-3.1-Tulu-3-8B"
TEMPLATE_ID = "templ-5"
TEMPLATES_CSV = "stance_judge/adapted_issuebench_stance_prompt_templates.csv"
BATCH_SIZE = 16
MAX_NEW_TOKENS = 64
CHUNK_ROWS = 200_000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input_final_csv")
    p.add_argument("output_csv")
    p.add_argument("n_per_template", nargs="?", type=int, default=20)
    p.add_argument("--judge-model", default=JUDGE_MODEL)
    p.add_argument("--template-id", default=TEMPLATE_ID)
    p.add_argument("--templates-csv", default=TEMPLATES_CSV)
    p.add_argument("--backend", default="transformers", choices=("transformers", "vllm"))
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--sample-size", type=int, default=None)
    p.add_argument("--chat-template-kwargs", type=json.loads, default=None,
                   help='JSON forwarded to apply_chat_template, e.g. \'{"enable_thinking": false}\'')
    return p.parse_args()


def sample_uniform(csv_path, n, seed=0):
    """Uniform random sample of n rows from a CSV too big to load whole:
    tag every row with a random key chunk by chunk and keep the n smallest
    keys seen so far. (Line-based sampling isn't safe -- responses contain
    embedded newlines.)"""
    rng = np.random.default_rng(seed)
    kept = None
    for chunk in pd.read_csv(csv_path, chunksize=CHUNK_ROWS):
        chunk["_key"] = rng.random(len(chunk))
        kept = chunk if kept is None else pd.concat([kept, chunk])
        kept = kept.nsmallest(n, "_key")
    return kept.drop(columns="_key").reset_index(drop=True)


def main():
    args = parse_args()
    input_csv, output_csv, n_per_template = args.input_final_csv, args.output_csv, args.n_per_template

    if args.sample_size is not None:
        sampled = sample_uniform(input_csv, args.sample_size)
        print(f"Uniformly sampled {len(sampled)} rows; template_id counts:\n{sampled['template_id'].value_counts()}")
    else:
        df = pd.read_csv(input_csv)
        sampled = (
            df.groupby("template_id", group_keys=False)
            .apply(lambda g: g.sample(min(n_per_template, len(g)), random_state=0))
            .reset_index(drop=True)
        )
        print(f"Sampled {len(sampled)} rows across template_ids: {sorted(sampled['template_id'].unique())}")

    template = load_template(args.templates_csv, args.template_id)

    stances = sampled["stance"].fillna("").tolist()
    responses = sampled["response"].fillna("").tolist()

    if args.backend == "vllm":
        import vllm_backend
        model, tokenizer = vllm_backend.load_judge_model(
            args.judge_model, tensor_parallel_size=args.tensor_parallel_size
        )
        raw_outputs = vllm_backend.classify_stance_batch(
            model, tokenizer, template, stances, responses, max_new_tokens=MAX_NEW_TOKENS,
            chat_template_kwargs=args.chat_template_kwargs,
        )
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Loading judge model {args.judge_model} on {device}")
        tokenizer = AutoTokenizer.from_pretrained(args.judge_model, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(args.judge_model, device_map=device, dtype=torch.float16)
        model.eval()

        raw_outputs = []
        for i in range(0, len(sampled), BATCH_SIZE):
            batch_stances = stances[i:i + BATCH_SIZE]
            batch_responses = responses[i:i + BATCH_SIZE]
            texts = [template.format(stance=s, response_text=r) for s, r in zip(batch_stances, batch_responses)]
            messages = [[{"role": "user", "content": t}] for t in texts]

            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt", padding=True,
                **(args.chat_template_kwargs or {}),
            ).to(model.device)

            outputs = model.generate(
                **inputs, do_sample=False, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tokenizer.eos_token_id,
            )
            prompt_len = inputs["input_ids"].shape[1]
            batch_raw = [tokenizer.decode(o[prompt_len:], skip_special_tokens=True) for o in outputs]
            raw_outputs.extend(batch_raw)
            print(f"  {min(i + BATCH_SIZE, len(sampled))}/{len(sampled)}")

    sampled["judge_raw_output"] = raw_outputs
    sampled["judge_reparsed_label"] = [parse_label(r) for r in raw_outputs]

    has_prior_run = "stance_pred" in sampled.columns
    if has_prior_run:
        sampled["reproduced_original"] = sampled["judge_reparsed_label"] == sampled["stance_pred"]
        keep_cols = [
            "prompt_id", "template_id", "stance", "response", "stance_pred",
            "judge_raw_output", "judge_reparsed_label", "reproduced_original",
        ]
    else:
        keep_cols = ["prompt_id", "template_id", "stance", "response", "judge_raw_output", "judge_reparsed_label"]
    sampled[keep_cols].to_csv(output_csv, index=False)
    print(f"Saved {len(sampled)} rows to {output_csv}")
    n_unparseable = sampled["judge_reparsed_label"].isna().sum()
    print(f"{n_unparseable}/{len(sampled)} judge outputs didn't match a recognized label")
    print(sampled["judge_reparsed_label"].value_counts(dropna=False).to_string())
    if has_prior_run:
        n_mismatch = (~sampled["reproduced_original"]).sum()
        print(f"{n_mismatch}/{len(sampled)} rows did NOT reproduce the original stance_pred under identical (greedy) settings")


if __name__ == "__main__":
    main()
