"""
Split a generations final.csv into N chunk directories, each holding its own
final.csv slice plus a classify_stance.py-compatible config.yml -- so the
full stance-classification run can be parallelized as a SLURM job array
(see run_stance_chunk.sbatch) instead of one long, non-resumable job. A
crash only costs one chunk's worth of work, not the whole run.

Each chunk dir gets <base_dir>/chunk_XX/final.csv (a literal slice of the
input, header + its rows) and <base_dir>/chunk_XX/config.yml. The config's
`model`/`sampling_configs`/`prompts` fields are unused placeholders --
classify_stance.py's config loader requires them present (see
pipeline_config.py) but never reads them; only `stance_classifier` and
`output_dir` matter here.

Usage:
    python split_for_stance_chunks.py <input_final_csv> <base_dir> \\
        --n-chunks 11 \\
        --judge-model /path/to/judge \\
        --template-id templ-10 \\
        [--backend vllm] [--tensor-parallel-size 2] \\
        [--chat-template-kwargs '{"enable_thinking": false}']
"""

import argparse
import csv
import math
import os
import sys

import yaml

csv.field_size_limit(sys.maxsize)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input_csv")
    p.add_argument("base_dir")
    p.add_argument("--n-chunks", type=int, required=True)
    p.add_argument("--judge-model", required=True)
    p.add_argument("--template-id", required=True)
    p.add_argument("--templates-csv", default="stance_judge/adapted_issuebench_stance_prompt_templates.csv")
    p.add_argument("--backend", default="vllm", choices=("transformers", "vllm"))
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--chat-template-kwargs", default=None, help='JSON string, e.g. \'{"enable_thinking": false}\'')
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()

    # Two streaming passes (count, then write) instead of list(reader) --
    # a multi-GB final.csv loaded whole as Python objects can exceed a
    # login node's per-process memory limit even when the system overall
    # has plenty free (hit this directly: 137/SIGKILL on an 891k-row,
    # 5.77GB file, no MemoryError -- the OS killed it, Python never saw it).
    with open(args.input_csv, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        n_rows = sum(1 for _ in reader)
    chunk_size = math.ceil(n_rows / args.n_chunks)
    print(f"{n_rows} rows -> {args.n_chunks} chunks of up to {chunk_size} rows each")

    os.makedirs(args.base_dir, exist_ok=True)

    stance_classifier = {
        "model": args.judge_model,
        "template_id": args.template_id,
        "templates_csv": args.templates_csv,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "backend": args.backend,
        "tensor_parallel_size": args.tensor_parallel_size,
    }
    if args.chat_template_kwargs:
        import json
        stance_classifier["chat_template_kwargs"] = json.loads(args.chat_template_kwargs)

    # Write every chunk's config.yml up front (cheap, no row data needed),
    # then stream the input once, rotating to the next chunk's writer every
    # chunk_size rows -- never more than one row in memory at a time.
    chunk_dirs = []
    for i in range(args.n_chunks):
        chunk_dir = os.path.join(args.base_dir, f"chunk_{i:02d}")
        os.makedirs(chunk_dir, exist_ok=True)
        chunk_dirs.append(chunk_dir)

        config = {
            "model": "unused -- classify_stance.py doesn't read this, only pipeline_config.load_config() requires the key",
            "sampling_configs": [],
            "prompts": {"dataset_csv": "unused.csv"},
            "stance_classifier": stance_classifier,
            "output_dir": chunk_dir,
            "resume": True,
        }
        with open(os.path.join(chunk_dir, "config.yml"), "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

    counts = [0] * args.n_chunks
    with open(args.input_csv, newline="") as f_in:
        reader = csv.reader(f_in)
        next(reader)  # header, already handled above

        chunk_idx = -1
        writer = f_out = None
        rows_in_chunk = chunk_size  # forces opening chunk 0 on the first row
        for row in reader:
            if rows_in_chunk == chunk_size and chunk_idx + 1 < args.n_chunks:
                if f_out:
                    f_out.close()
                chunk_idx += 1
                f_out = open(os.path.join(chunk_dirs[chunk_idx], "final.csv"), "w", newline="")
                writer = csv.writer(f_out)
                writer.writerow(header)
                rows_in_chunk = 0
            writer.writerow(row)
            rows_in_chunk += 1
            counts[chunk_idx] += 1
        if f_out:
            f_out.close()

    for i, chunk_dir in enumerate(chunk_dirs):
        print(f"  chunk_{i:02d}: {counts[i]} rows -> {chunk_dir}")

    print(f"Done. {args.n_chunks} chunk configs ready under {args.base_dir}")


if __name__ == "__main__":
    main()
