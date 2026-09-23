"""
Generate model responses to a set of prompts and save them to a CSV.

This covers the generation part of generative_eval.ipynb: given a pipeline
config (see pipeline_config.py's docstring), build/load the prompt set, run
generation for each (prompt, sampling config) pair with the configured
model, and write the raw generations to <output_dir>/generations.csv (also
writing the resolved prompt set to <output_dir>/prompts.csv for
provenance). Everything downstream (stance classification, refusal
classification) is left to classify_stance.py / classify_refusal.py.

Two generation backends, picked via `generation.backend` in the config
(default "transformers"):

    generation:
      max_new_tokens: 300
      backend: transformers   # one model.generate() call per (prompt,
                               #  sampling config), streamed to CSV as it
                               #  goes -- this module's own implementation.
      # backend: vllm         # every pending (prompt, sampling config)
                               #  request batched into one call via vLLM's
                               #  continuous batching -- see vllm_backend.py
                               #  for the speed/resume trade-off. Requires
                               #  `vllm` installed; not needed otherwise.

Usage:
    python generate_texts.py config.yml
"""

import csv
import os
import sys

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from generation_common import fieldnames_for, load_completed_pairs, model_own_sampling_overrides, resolve_sampling_kwargs
from pipeline_config import load_config, output_path
from prompts import resolve_prompt_df


def load_model(model_path):
    name = model_path.rstrip("/").split("/")[-1]
    print(f"Loading {name}...")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    default_overrides = model_own_sampling_overrides(model_path)
    print(f"  empty-params override: {default_overrides or '(uses model own generation_config)'}")
    return name, model, tokenizer, default_overrides


def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens=500,
    do_sample=True,
    num_generations=1,
    temperature=None,
    top_p=None,
    top_k=None,
    repetition_penalty=None,
):
    """Generate `num_generations` samples for `prompt`.

    temperature/top_p/top_k/repetition_penalty default to None, meaning
    "don't override -- use whatever's in this model's own generation_config"
    (transformers merges any kwargs left unset with model.generation_config).
    Callers resolve the actual values per sampling config before calling this.
    """
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    sampling_kwargs = {
        k: v for k, v in {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
        }.items() if v is not None
    }

    outputs = model.generate(
        **inputs,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_return_sequences=num_generations,
        **sampling_kwargs,
    )

    prompt_len = inputs["input_ids"].shape[-1]
    return [
        tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
        for output in outputs
    ]


def run_generation(
    model_name,
    model,
    tokenizer,
    df,
    max_new_tokens,
    sampling_configs,
    output_csv,
    model_default_overrides,
    resume=True,
):
    """Generate responses under every sampling config and stream each row to
    output_csv as soon as it's produced.

    sampling_configs: list of {"name", "params", "num_generations"} dicts.

    If resume=True and output_csv already has complete batches for some
    (prompt_id, model, config) triples, those are kept as-is and skipped
    instead of being regenerated.
    """
    fieldnames = fieldnames_for(df)
    metadata_cols = [c for c in df.columns if c not in ("prompt_id", "prompt")]
    num_generations_by_config = {c["name"]: c.get("num_generations", 1) for c in sampling_configs}

    completed, kept = (
        (set(), None) if not resume else load_completed_pairs(output_csv, num_generations_by_config)
    )

    if kept is not None:
        kept.to_csv(output_csv, index=False, columns=fieldnames)
        print(f"Resuming: {len(completed)} (prompt, model, config) batches already complete, keeping {len(kept)} rows.")
        file_mode = "a"
    else:
        file_mode = "w"

    n_written = 0
    with open(output_csv, file_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if file_mode == "w":
            writer.writeheader()

        rows = list(df.to_dict("records"))
        total = len(rows) * len(sampling_configs)
        pbar = tqdm(total=total, desc="Generating...")
        pbar.update(len(completed))

        for row in rows:
            prompt_id = row["prompt_id"]
            p = row["prompt"]
            metadata = {c: row[c] for c in metadata_cols}

            for sc in sampling_configs:
                config_name = sc["name"]
                if (prompt_id, model_name, config_name) in completed:
                    continue

                sampling_kwargs = resolve_sampling_kwargs(sc.get("params", {}), model_default_overrides)
                generations = generate_text(
                    model,
                    tokenizer,
                    p,
                    do_sample=True,
                    max_new_tokens=max_new_tokens,
                    num_generations=num_generations_by_config[config_name],
                    **sampling_kwargs,
                )

                for generation_id, response in enumerate(generations):
                    writer.writerow({
                        "prompt_id": prompt_id,
                        "prompt": p,
                        **metadata,
                        "model": model_name,
                        "config": config_name,
                        "generation_id": generation_id,
                        "response": response,
                    })
                    n_written += 1
                f.flush()
                pbar.update(1)

        pbar.close()

    return n_written


def backend_module(name):
    if name == "transformers":
        return sys.modules[__name__]
    if name == "vllm":
        import vllm_backend
        return vllm_backend
    raise ValueError(f"Unknown generation.backend: {name!r} (expected 'transformers' or 'vllm')")


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: python {sys.argv[0]} config.yml")
    cfg = load_config(sys.argv[1])

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    os.makedirs(cfg["output_dir"], exist_ok=True)

    df = resolve_prompt_df(cfg["prompts"])
    df.to_csv(output_path(cfg, "prompts.csv"), index=False)

    generation_cfg = cfg.get("generation", {})
    backend = backend_module(generation_cfg.get("backend", "transformers"))

    model_name, model, tokenizer, model_default_overrides = backend.load_model(cfg["model"])

    n_written = backend.run_generation(
        model_name, model, tokenizer, df,
        max_new_tokens=generation_cfg.get("max_new_tokens", 300),
        sampling_configs=cfg["sampling_configs"],
        output_csv=output_path(cfg, "generations.csv"),
        model_default_overrides=model_default_overrides,
        resume=cfg.get("resume", True),
    )

    print(f"Saved {n_written} generations to {output_path(cfg, 'generations.csv')}")


if __name__ == "__main__":
    main()
