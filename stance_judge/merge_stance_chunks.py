"""
Merge completed chunk_XX/final.csv files (from split_for_stance_chunks.py +
run_stance_chunk.sbatch) back into one final.csv with stance_pred filled in
for every row. Verifies completeness first -- refuses to merge if any
chunk's final.csv is missing the stance_pred column (not yet processed) or
if the total row count doesn't match the original input, so a partial run
can't silently produce a merged file with gaps.

Usage:
    python merge_stance_chunks.py <base_dir> <original_input_csv> <output_csv>
"""

import csv
import glob
import os
import sys

csv.field_size_limit(sys.maxsize)


def main():
    if len(sys.argv) != 4:
        raise SystemExit(f"Usage: python {sys.argv[0]} base_dir original_input_csv output_csv")
    base_dir, original_csv, output_csv = sys.argv[1:4]

    chunk_dirs = sorted(glob.glob(os.path.join(base_dir, "chunk_*")))
    if not chunk_dirs:
        raise SystemExit(f"No chunk_* directories found under {base_dir}")

    with open(original_csv, newline="") as f:
        n_expected = sum(1 for _ in csv.reader(f)) - 1  # minus header

    all_rows, fieldnames, missing = [], None, []
    for d in chunk_dirs:
        chunk_csv = os.path.join(d, "final.csv")
        if not os.path.exists(chunk_csv):
            missing.append((d, "no final.csv"))
            continue
        with open(chunk_csv, newline="") as f:
            reader = csv.DictReader(f)
            if "stance_pred" not in reader.fieldnames:
                missing.append((d, "final.csv has no stance_pred column yet"))
                continue
            fieldnames = fieldnames or reader.fieldnames
            all_rows.extend(reader)

    if missing:
        print(f"NOT merging -- {len(missing)} chunk(s) incomplete:")
        for d, why in missing:
            print(f"  {d}: {why}")
        raise SystemExit(1)

    if len(all_rows) != n_expected:
        raise SystemExit(f"NOT merging -- expected {n_expected} rows from {original_csv}, got {len(all_rows)} across chunks")

    n_unparseable = sum(1 for r in all_rows if not r.get("stance_pred"))
    print(f"All {len(chunk_dirs)} chunks complete, {len(all_rows)} rows total, {n_unparseable} unparseable stance_pred")

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote merged file to {output_csv}")


if __name__ == "__main__":
    main()
