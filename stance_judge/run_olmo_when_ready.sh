#!/bin/bash
# Prepared ahead of the OLMo-2-7B / OLMo-2-7B-NR generations landing.
# Chunks + submits gemma-4-31B-it/templ-10 stance classification for both,
# using the same settings validated on the other 8 model files (12 chunks,
# 4x H100, ata@h100 -- see stance_judge/chunk_stance_run.sh).
#
# Safe to re-run: if a data file isn't there yet, it's skipped with a
# message rather than erroring, so you can just re-run this once the
# generations are ready instead of re-typing the commands from memory.
#
# Usage (from the project root): ./stance_judge/run_olmo_when_ready.sh

set -uo pipefail
cd /lustre/fswork/projects/rech/oag/unz84ar/programs/generative_eval
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_one() {
    local MODEL_DIR="$1" BASE_DIR="stance_chunks_$1"
    local CSV="data/$MODEL_DIR/final.csv"
    if [ ! -f "$CSV" ]; then
        CSV="data/$MODEL_DIR/generations.csv"  # refusal classification may not have run yet
    fi
    if [ ! -f "$CSV" ]; then
        echo "Not ready yet: no final.csv or generations.csv under data/$MODEL_DIR -- skipping"
        return
    fi
    if [ -d "$BASE_DIR" ]; then
        echo "Already chunked: $BASE_DIR exists -- skipping (delete it first to rebuild)"
        return
    fi
    echo "=== Chunking + submitting: $CSV -> $BASE_DIR ==="
    "$HERE/chunk_stance_run.sh" "$CSV" "$BASE_DIR" 12 4 ata@h100
}

run_one "leolabat--OLMo-2-7B-SFT-100k"
run_one "leolabat--OLMo-2-7B-SFT-100k-NR"
