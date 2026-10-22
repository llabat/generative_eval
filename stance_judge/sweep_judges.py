"""
Run one judge model over every candidate template against the hand-labeled
gold set (see build_gold_set.py), so template variants can compete for one
model without reloading it between templates. Sweeping multiple MODELS is
left to separate sbatch jobs (one per model) that all append to the same
--output-csv, since each judge model typically needs its own GPU count/QoS
anyway; score_judges.py then reads across every model x template combo in
that shared file.

Output is long-format: one row per (model, template_id, gold row), with
columns model, template_id, row_number, gold_labels, judge_raw_output,
judge_reparsed_label. Resumable: if --output-csv already exists, template_ids
that already have a full set of rows for this model are skipped.

CAUTION: the append/resume logic is safe for SEQUENTIAL reruns of this
script against the same --output-csv (e.g. rerunning a model later to add
more templates), but NOT for two invocations running CONCURRENTLY against
the same file (e.g. several sbatch jobs for different models launched at
once) -- there's no file locking, so concurrent appends can interleave or
duplicate the header. Give concurrent jobs separate --output-csv files
instead, and pass all of them (comma-separated) to score_judges.py.

Usage:
    python sweep_judges.py gold_stance_500.csv sweep_results.csv \\
        --judge-model /path/to/judge \\
        --templates templ-5,templ-13,templ-17 \\
        [--backend vllm --tensor-parallel-size 4]
        [--chat-template-kwargs '{"enable_thinking": false}']
"""

import argparse
import csv
import json
import os
import sys

csv.field_size_limit(sys.maxsize)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classify_stance import DEFAULT_TEMPLATES_CSV, backend_module, load_template, parse_label

MAX_NEW_TOKENS = 64
BATCH_SIZE = 16  # transformers backend only


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("gold_csv")
    p.add_argument("output_csv")
    p.add_argument("--judge-model", required=True)
    p.add_argument("--templates", required=True, help='comma-separated template_ids, or "all" for every id in --templates-csv')
    p.add_argument("--templates-csv", default=DEFAULT_TEMPLATES_CSV)
    p.add_argument("--backend", default="transformers", choices=("transformers", "vllm"))
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--chat-template-kwargs", type=json.loads, default=None,
                    help='JSON forwarded to apply_chat_template, e.g. \'{"enable_thinking": false}\'')
    p.add_argument("--temperature", type=float, default=None,
                    help="None (default) -> greedy/deterministic. A float (e.g. 1.0) -> sampling at that temperature.")
    p.add_argument("--n-samples", type=int, default=1,
                    help="Repeat each template this many times (only meaningful with --temperature set -- "
                         "greedy is deterministic, so repeating it would just waste compute on identical output).")
    args = p.parse_args()
    if args.n_samples > 1 and args.temperature is None:
        p.error("--n-samples > 1 requires --temperature (greedy decoding is deterministic, repeats would be identical)")
    return args


def resolve_template_ids(spec, templates_csv):
    if spec != "all":
        return spec.split(",")
    with open(templates_csv, newline="") as f:
        return [r["id"] for r in csv.DictReader(f)]


def already_done(output_csv, model_name, n_gold, n_samples):
    if not os.path.exists(output_csv):
        return set()
    counts = {}
    with open(output_csv, newline="") as f:
        for r in csv.DictReader(f):
            if r["model"] == model_name:
                counts[r["template_id"]] = counts.get(r["template_id"], 0) + 1
    needed = n_gold * n_samples
    return {tid for tid, n in counts.items() if n == needed}


def main():
    args = parse_args()
    model_name = args.judge_model.rstrip("/").split("/")[-1]

    gold_rows = list(csv.DictReader(open(args.gold_csv, newline="")))
    stances = [r["stance"] for r in gold_rows]
    responses = [r["response"] for r in gold_rows]
    print(f"Loaded {len(gold_rows)} gold rows")

    template_ids = resolve_template_ids(args.templates, args.templates_csv)
    done = already_done(args.output_csv, model_name, len(gold_rows), args.n_samples)
    pending = [t for t in template_ids if t not in done]
    print(f"Templates: {len(template_ids)} total, {len(done)} already done for {model_name}, {len(pending)} pending: {pending}")
    if not pending:
        return

    backend = backend_module(args.backend)
    print(f"Loading judge model {args.judge_model} (backend={args.backend}, tensor_parallel_size={args.tensor_parallel_size})")
    model, tokenizer = backend.load_judge_model(args.judge_model, tensor_parallel_size=args.tensor_parallel_size)

    write_header = not os.path.exists(args.output_csv)
    fieldnames = ["model", "template_id", "sample_idx", "temperature", "row_number", "gold_labels", "judge_raw_output", "judge_reparsed_label"]
    with open(args.output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for template_id in pending:
            template = load_template(args.templates_csv, template_id)
            print(f"--- {template_id} ---")
            for sample_idx in range(args.n_samples):
                kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, temperature=args.temperature, chat_template_kwargs=args.chat_template_kwargs)
                if args.backend == "transformers":
                    kwargs["batch_size"] = BATCH_SIZE
                raw_outputs = backend.classify_stance_batch(model, tokenizer, template, stances, responses, **kwargs)

                for gold_row, raw in zip(gold_rows, raw_outputs):
                    writer.writerow({
                        "model": model_name,
                        "template_id": template_id,
                        "sample_idx": sample_idx,
                        "temperature": args.temperature if args.temperature is not None else 0.0,
                        "row_number": gold_row["row_number"],
                        "gold_labels": gold_row["gold_labels"],
                        "judge_raw_output": raw,
                        "judge_reparsed_label": parse_label(raw),
                    })
                f.flush()
                print(f"  sample {sample_idx}: wrote {len(gold_rows)} rows")

    print(f"Done. Results in {args.output_csv}")


if __name__ == "__main__":
    main()
