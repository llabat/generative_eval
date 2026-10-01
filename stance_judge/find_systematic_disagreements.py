"""
One-off diagnostic: find gold rows where the judges (pooled across every
model x template combo in the given sweep result files) confidently agree
on a label that ISN'T in the gold annotation. Meant for re-annotation review
-- see the 2026-09-30 conversation: 45/500 rows had >=75% of 48 judge calls
(3 models x 16 templates) converging on a single non-gold label, almost all
of them one notch less extreme than the human label (1->2, 2->3, 4->3),
plus one row the other way (3->4) -- worth a second annotation pass with
that pattern in mind, rather than assuming either side is simply "right".

Usage:
    python find_systematic_disagreements.py gold_stance_500.csv output.csv \\
        sweep_results_a.csv sweep_results_b.csv ... [--min-consensus 0.75]
"""

import argparse
import collections
import csv
import sys

csv.field_size_limit(sys.maxsize)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("gold_csv")
    p.add_argument("output_csv")
    p.add_argument("results_csvs", nargs="+")
    p.add_argument("--min-consensus", type=float, default=0.75)
    return p.parse_args()


def main():
    args = parse_args()

    gold = {r["row_number"]: r for r in csv.DictReader(open(args.gold_csv, newline=""))}

    preds = collections.defaultdict(list)
    for path in args.results_csvs:
        for r in csv.DictReader(open(path, newline="")):
            preds[r["row_number"]].append(r["judge_reparsed_label"] or "none")

    out_rows = []
    for rn, plist in preds.items():
        g = gold[rn]
        gold_set = set(g["gold_labels"].split(";"))
        counts = collections.Counter(plist)
        consensus_label, consensus_n = counts.most_common(1)[0]
        consensus_frac = consensus_n / len(plist)
        if consensus_label in gold_set or consensus_frac < args.min_consensus:
            continue
        out_rows.append({
            "row_number": rn,
            "stance": g["stance"],
            "response": g["response"],
            "gold_raw_label": g["raw_label"],
            "gold_labels": g["gold_labels"],
            "judge_consensus_label": consensus_label,
            "judge_consensus_frac": f"{consensus_frac:.3f}",
            "n_judge_calls": len(plist),
            "new_label": "",
        })

    out_rows.sort(key=lambda r: -float(r["judge_consensus_frac"]))

    fieldnames = ["row_number", "stance", "response", "gold_raw_label", "gold_labels",
                  "judge_consensus_label", "judge_consensus_frac", "n_judge_calls", "new_label"]
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows (consensus >= {args.min_consensus}) to {args.output_csv}")


if __name__ == "__main__":
    main()
