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

Usage:
    python diagnose_stance_judge.py <input_final_csv> <output_csv> [n_per_template]
"""

import sys

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from classify_stance import load_template, parse_label

JUDGE_MODEL = "allenai/Llama-3.1-Tulu-3-8B"
TEMPLATE_ID = "templ-5"
TEMPLATES_CSV = "adapted_issuebench_stance_prompt_templates.csv"
BATCH_SIZE = 16
MAX_NEW_TOKENS = 64


def main():
    if len(sys.argv) not in (3, 4):
        raise SystemExit(f"Usage: python {sys.argv[0]} input_final_csv output_csv [n_per_template]")
    input_csv, output_csv = sys.argv[1], sys.argv[2]
    n_per_template = int(sys.argv[3]) if len(sys.argv) == 4 else 20

    df = pd.read_csv(input_csv)
    sampled = (
        df.groupby("template_id", group_keys=False)
        .apply(lambda g: g.sample(min(n_per_template, len(g)), random_state=0))
        .reset_index(drop=True)
    )
    print(f"Sampled {len(sampled)} rows across template_ids: {sorted(sampled['template_id'].unique())}")

    template = load_template(TEMPLATES_CSV, TEMPLATE_ID)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading judge model {JUDGE_MODEL} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(JUDGE_MODEL, device_map=device, dtype=torch.float16)
    model.eval()

    stances = sampled["stance"].fillna("").tolist()
    responses = sampled["response"].fillna("").tolist()

    raw_outputs = []
    for i in range(0, len(sampled), BATCH_SIZE):
        batch_stances = stances[i:i + BATCH_SIZE]
        batch_responses = responses[i:i + BATCH_SIZE]
        texts = [template.format(stance=s, response_text=r) for s, r in zip(batch_stances, batch_responses)]
        messages = [[{"role": "user", "content": t}] for t in texts]

        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", padding=True,
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
    sampled["reproduced_original"] = sampled["judge_reparsed_label"] == sampled["stance_pred"]

    keep_cols = [
        "prompt_id", "template_id", "stance", "response", "stance_pred",
        "judge_raw_output", "judge_reparsed_label", "reproduced_original",
    ]
    sampled[keep_cols].to_csv(output_csv, index=False)
    n_mismatch = (~sampled["reproduced_original"]).sum()
    print(f"Saved {len(sampled)} rows to {output_csv}")
    print(f"{n_mismatch}/{len(sampled)} rows did NOT reproduce the original stance_pred under identical (greedy) settings")


if __name__ == "__main__":
    main()
