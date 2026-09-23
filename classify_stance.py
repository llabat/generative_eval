"""
Run the LLM-as-judge stance classifier over a pipeline's generations and add
a `stance_pred` column: a 1-5 Likert label (1 = text exclusively argues for
the stance, 5 = exclusively against it) or "refusal", per one of the
IssueBench-derived templates in adapted_issuebench_stance_prompt_templates.csv
(adapted from Rottger et al.'s IssueBench stance-classifier evaluation).

Modular by design -- template_id and the judge model are both config-driven,
not hardcoded, so you can sweep template_id (see that CSV's `note` column
for what each variant tests -- text/task first, static vs dynamic, llm
framing, negative examples, ...) against a small labeled set before settling
on one for real runs. Every run's `stance_template_id` / `stance_judge_model`
columns record which choice produced its `stance_pred`, so results from
different sweeps stay distinguishable when concatenating runs -- named with
a `stance_` prefix, not `template_id`/`model`, because the generations
dataframe already carries columns of those exact names for a different
thing (the neutral/support/oppose *generation* template and the *generation*
model -- see prompts.py / generate_texts.py). Same reasoning as the existing
`stance` (proposition text, from opinions_csv) vs `stance_pred` split.

Requires a `stance_classifier:` section:

    stance_classifier:
      model: /path/to/judge/model
      template_id: templ-6          # id from templates_csv, required
      templates_csv: adapted_issuebench_stance_prompt_templates.csv  # optional, default shown
      batch_size: 64                 # optional, default 64
      max_new_tokens: 64             # optional, default 64 -- matches IssueBench's
                                       #  own budget for this judge task; not every
                                       #  model answers with just the bare label, so
                                       #  parse_label() below allows for a completion
                                       #  that reasons before answering

Reads <output_dir>/final.csv if it already exists (e.g. classify_refusal.py
ran first), else <output_dir>/generations.csv, and writes/overwrites
<output_dir>/final.csv -- so classify_stance.py and classify_refusal.py can
run in either order, or just one of them, without clobbering each other's
column.

Usage:
    python classify_stance.py config.yml
"""

import os
import re
import sys

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipeline_config import load_config, output_path

DEFAULT_TEMPLATES_CSV = "adapted_issuebench_stance_prompt_templates.csv"
VALID_LABELS = {"1", "2", "3", "4", "5", "refusal"}
LABEL_RE = re.compile(r"\b(1|2|3|4|5|refusal)\b", re.IGNORECASE)


def load_template(templates_csv, template_id):
    templates_df = pd.read_csv(templates_csv)
    for col in ("id", "adapted_template"):
        if col not in templates_df.columns:
            raise ValueError(f"{templates_csv} must have an '{col}' column")

    matches = templates_df[templates_df["id"] == template_id]
    if matches.empty:
        available = ", ".join(templates_df["id"])
        raise ValueError(
            f"template_id {template_id!r} not found in {templates_csv} (available: {available})"
        )
    return matches.iloc[0]["adapted_template"]


def parse_label(raw):
    """Normalize a judge completion down to one of VALID_LABELS, or None if
    it doesn't contain anything recognizable (empty generation, or genuinely
    off-script text) -- so a malformed completion shows up as a countable
    None rather than a silently-wrong label.

    Fast path: the whole (stripped/unquoted) completion is already just the
    label -- the common case when the judge follows the "answer with just
    ..." instruction. Fallback: search for the last standalone label token
    in the completion, since max_new_tokens=64 (matching IssueBench's own
    budget for this task) is sized to allow the judge to reason before
    answering, not every model does answer with just the bare label.
    """
    cleaned = raw.strip().strip("\"'").strip().rstrip(".:,;").lower()
    if cleaned in VALID_LABELS:
        return cleaned

    matches = LABEL_RE.findall(raw)
    return matches[-1].lower() if matches else None


def classify_stance_batch(model, tokenizer, template, stances, responses, batch_size=64, max_new_tokens=64):
    results = []
    pbar = tqdm(total=len(stances), desc="Classifying stance")

    for i in range(0, len(stances), batch_size):
        batch_stances = stances[i:i + batch_size]
        batch_responses = responses[i:i + batch_size]

        texts = [
            template.format(stance=stance, response_text=response)
            for stance, response in zip(batch_stances, batch_responses)
        ]

        messages = [[{"role": "user", "content": text}] for text in texts]

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

        prompt_len = inputs["input_ids"].shape[1]

        batch_results = [
            tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
            for output in outputs
        ]

        results.extend(batch_results)
        pbar.update(len(batch_stances))

    pbar.close()
    return results


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: python {sys.argv[0]} config.yml")
    cfg = load_config(sys.argv[1])

    if "stance_classifier" not in cfg:
        raise SystemExit(f"{sys.argv[1]} has no stance_classifier section -- nothing to do")
    sc_cfg = cfg["stance_classifier"]

    if "template_id" not in sc_cfg:
        raise ValueError(f"{sys.argv[1]}: stance_classifier.template_id is required")
    template_id = sc_cfg["template_id"]
    templates_csv = sc_cfg.get("templates_csv", DEFAULT_TEMPLATES_CSV)
    template = load_template(templates_csv, template_id)
    print(f"Using stance template {template_id!r} from {templates_csv}")

    final_csv = output_path(cfg, "final.csv")
    generations_csv = output_path(cfg, "generations.csv")
    input_csv = final_csv if os.path.exists(final_csv) else generations_csv

    print(f"Loading {input_csv}")
    df = pd.read_csv(input_csv)
    for col in ("stance", "response"):
        if col not in df.columns:
            raise ValueError(f"Expected a '{col}' column in {input_csv}")

    model_path = sc_cfg["model"]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading judge model {model_path} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, dtype=torch.float16)
    model.eval()

    raw = classify_stance_batch(
        model,
        tokenizer,
        template,
        df["stance"].fillna("").tolist(),
        df["response"].fillna("").tolist(),
        batch_size=sc_cfg.get("batch_size", 64),
        max_new_tokens=sc_cfg.get("max_new_tokens", 64),
    )

    parsed = [parse_label(r) for r in raw]
    n_unparseable = sum(p is None for p in parsed)
    print(f"{n_unparseable}/{len(parsed)} judge outputs didn't match a recognized label ({sorted(VALID_LABELS)})")

    df["stance_pred"] = parsed
    df["stance_score"] = pd.to_numeric(df["stance_pred"], errors="coerce")  # NaN for "refusal"/unparseable
    df["stance_template_id"] = template_id
    df["stance_judge_model"] = model_path

    df.to_csv(final_csv, index=False)
    print(f"Saved {len(df)} rows with stance predictions to {final_csv}")


if __name__ == "__main__":
    main()
