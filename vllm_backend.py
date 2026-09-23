"""
vLLM generation backend for generate_texts.py, selected via
`generation.backend: vllm` in a pipeline config. Kept as its own module so
the default transformers backend (generate_texts.py) never needs `vllm`
importable -- it's imported lazily here, only when this backend is chosen.

Why this exists: the transformers backend issues one model.generate() call
per (prompt, sampling config) pair, sequentially -- fine for a handful of
prompts, but for a real sweep (dozens of prompts x several sampling
configs x large N) that's hundreds of sequential calls with no batching
across prompts. vLLM's offline LLM.generate() takes every pending request
at once and batches them together via continuous batching + PagedAttention,
which is generally far higher throughput for exactly this kind of
"generate a lot of samples, no interactivity" workload.

Trade-off vs. the transformers backend: results are only written to CSV
once the whole batched call returns -- vLLM's offline API doesn't stream
completions back as individual requests finish -- so a crash mid-call
loses that whole call's progress rather than just the in-flight request.
Rerunning with resume: true (the default) just redoes it; there's no
partial-batch resumability the way the transformers backend's per-request
flushing gives you.

Caveat worth checking once you have vLLM available: whether your installed
version merges a sampling config's unset fields (e.g. `params: {}`'s
FALLBACK_SAMPLING, which never sets top_k) with the model's own
generation_config.json the same way transformers does, or falls back to
vLLM's own SamplingParams defaults instead. If you see different-looking
outputs than the transformers backend for the same "model_default"
config, this is the first thing to check.
"""

import csv

from tqdm import tqdm
from transformers import AutoTokenizer

from generation_common import fieldnames_for, load_completed_pairs, model_own_sampling_overrides, resolve_sampling_kwargs


def load_model(model_path):
    from vllm import LLM  # lazy: only this backend needs vllm installed

    name = model_path.rstrip("/").split("/")[-1]
    print(f"Loading {name} with vLLM...")
    llm = LLM(model=model_path, dtype="float16")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    default_overrides = model_own_sampling_overrides(model_path)
    print(f"  empty-params override: {default_overrides or '(uses model own generation_config)'}")
    return name, llm, tokenizer, default_overrides


def run_generation(
    model_name,
    llm,
    tokenizer,
    df,
    max_new_tokens,
    sampling_configs,
    output_csv,
    model_default_overrides,
    resume=True,
):
    """Same signature/semantics as generate_texts.run_generation, but
    submits every pending (prompt, sampling config) request to vLLM as one
    batched call instead of looping model.generate() calls.

    sampling_configs: list of {"name", "params", "num_generations"} dicts.
    """
    from vllm import SamplingParams

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

    # Build every pending request up front (one request per (row, sampling
    # config) pair -- each request's own SamplingParams(n=num_generations)
    # gets that many samples back from vLLM in one go) so they can all be
    # batched together via continuous batching.
    pending_rows, pending_prompts, pending_params = [], [], []
    for row in df.to_dict("records"):
        prompt_id, p = row["prompt_id"], row["prompt"]
        for sc in sampling_configs:
            config_name = sc["name"]
            if (prompt_id, model_name, config_name) in completed:
                continue

            sampling_kwargs = resolve_sampling_kwargs(sc.get("params", {}), model_default_overrides)
            params = SamplingParams(
                n=num_generations_by_config[config_name],
                max_tokens=max_new_tokens,
                **sampling_kwargs,
            )

            messages = [{"role": "user", "content": p}]
            prompt_text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )

            pending_rows.append((row, config_name))
            pending_prompts.append(prompt_text)
            pending_params.append(params)

    n_written = 0
    if pending_prompts:
        print(f"Submitting {len(pending_prompts)} requests to vLLM...")
        outputs = llm.generate(pending_prompts, pending_params)

        with open(output_csv, file_mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if file_mode == "w":
                writer.writeheader()

            for (row, config_name), output in tqdm(
                zip(pending_rows, outputs), total=len(pending_rows), desc="Writing..."
            ):
                metadata = {c: row[c] for c in metadata_cols}
                for generation_id, completion in enumerate(output.outputs):
                    writer.writerow({
                        "prompt_id": row["prompt_id"],
                        "prompt": row["prompt"],
                        **metadata,
                        "model": model_name,
                        "config": config_name,
                        "generation_id": generation_id,
                        "response": completion.text,
                    })
                    n_written += 1
    elif file_mode == "w":
        # Nothing pending and nothing kept either (resume=False with an
        # empty prompt set, or all-empty sampling_configs) -- still leave a
        # valid, header-only CSV behind instead of no file at all.
        with open(output_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    return n_written
