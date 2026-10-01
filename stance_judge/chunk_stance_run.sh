#!/bin/bash
# Chunk a generations final.csv AND submit the stance-classification array
# job for it, in one command. Bakes in the winning judge (gemma-4-31B-it +
# templ-10, picked via the sweep -- see leaderboard_final.csv) and the GPU/
# account settings validated on the Llama-3.1-Tulu-3-8B-DPO run (4x H100,
# ata@h100, 23.4 GPU-hours for 891k rows) as defaults, all overridable.
#
# Usage:
#   ./chunk_stance_run.sh <input_final_csv> <base_dir> [n_chunks] [tensor_parallel_size] [account]
#
# n_chunks, if omitted, is auto-computed to target ~75000 rows/chunk.
# tensor_parallel_size defaults to 4; cpus-per-task is derived as TP*24
# (this cluster ties memory allocation to cpus-per-task, so this is also
# what sets the job's memory -- see the 480000M that worked for TP=4).
# account defaults to ata@h100.
#
# This builds the chunk dirs AND submits the array job. It does NOT merge
# results afterward -- once the array finishes, run:
#   python stance_judge/merge_stance_chunks.py <base_dir> <input_final_csv> <output_csv>
#
# Invoke this from the project root (generative_eval/), e.g.:
#   ./stance_judge/chunk_stance_run.sh data/<model>/final.csv stance_chunks_<name>
# <input_final_csv>/<base_dir> are resolved relative to wherever you run this
# from (NOT relative to this script's own location) -- it never cd's.

set -euo pipefail

INPUT_CSV="${1:?Usage: $0 <input_final_csv> <base_dir> [n_chunks] [tensor_parallel_size] [account]}"
BASE_DIR="${2:?Usage: $0 <input_final_csv> <base_dir> [n_chunks] [tensor_parallel_size] [account]}"
TP="${4:-4}"
ACCOUNT="${5:-ata@h100}"

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
echo "Submitting array job (gpu:$TP, cpus-per-task=$((TP * 24)), account=$ACCOUNT)..."
sbatch --gres=gpu:"$TP" --cpus-per-task=$((TP * 24)) --account="$ACCOUNT" \
    --array=0-$((N_CHUNKS - 1)) "$HERE/run_stance_chunk.sbatch" "$BASE_DIR"

echo
echo "Once all chunks finish, merge with:"
echo "  $VENV_PY $HERE/merge_stance_chunks.py $BASE_DIR $INPUT_CSV <output_csv>"
