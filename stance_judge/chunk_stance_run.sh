#!/bin/bash
# Thin wrapper around split_for_stance_chunks.py: bakes in the winning judge
# (gemma-4-31B-it + templ-10, picked via the sweep -- see leaderboard_final.csv)
# as defaults, and auto-sizes the chunk count from the row count, so a new
# source final.csv can be chunked with one short command instead of
# repeating the full judge/template/backend flags every time.
#
# Usage:
#   ./chunk_stance_run.sh <input_final_csv> <base_dir> [n_chunks] [tensor_parallel_size]
#
# n_chunks, if omitted, is auto-computed to target ~75000 rows/chunk --
# comfortable under the H100 dev QoS's 2h cap at ~15-16 rows/s on 2 GPUs.
# With more GPUs (tensor_parallel_size > 2) each chunk just finishes with
# more headroom, so the same target is fine regardless of GPU count.
# tensor_parallel_size defaults to 2.
#
# This only builds the chunk dirs -- it does NOT submit anything. It prints
# the sbatch command to run next.
#
# Invoke this from the project root (generative_eval/), e.g.:
#   ./stance_judge/chunk_stance_run.sh data/<model>/final.csv stance_chunks_<name>
# <input_final_csv>/<base_dir> are resolved relative to wherever you run this
# from (NOT relative to this script's own location) -- it never cd's.

set -euo pipefail

INPUT_CSV="${1:?Usage: $0 <input_final_csv> <base_dir> [n_chunks] [tensor_parallel_size]}"
BASE_DIR="${2:?Usage: $0 <input_final_csv> <base_dir> [n_chunks] [tensor_parallel_size]}"
TP="${4:-2}"

VENV_PY=/lustre/fsn1/projects/rech/oag/unz84ar/venvs/generative_eval/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -n "${3:-}" ]; then
    N_CHUNKS="$3"
else
    N_ROWS=$("$VENV_PY" -c "
import csv, sys
csv.field_size_limit(sys.maxsize)
with open('$INPUT_CSV', newline='') as f:
    print(sum(1 for _ in csv.reader(f)) - 1)
")
    N_CHUNKS=$(( (N_ROWS + 74999) / 75000 ))
    echo "Auto-sized: $N_ROWS rows -> $N_CHUNKS chunks (~75000 rows each)"
fi

"$VENV_PY" "$HERE/split_for_stance_chunks.py" \
    "$INPUT_CSV" "$BASE_DIR" \
    --n-chunks "$N_CHUNKS" \
    --judge-model /lustre/fsmisc/dataset/HuggingFace_Models/google/gemma-4-31B-it \
    --template-id templ-10 \
    --backend vllm \
    --tensor-parallel-size "$TP" \
    --chat-template-kwargs '{"enable_thinking": false}'

echo
echo "Next: submit the array job, e.g."
echo "  sbatch --gres=gpu:$TP --cpus-per-task=$((TP * 24)) --account=<account>@h100 --array=0-$((N_CHUNKS - 1)) stance_judge/run_stance_chunk.sbatch $BASE_DIR"
