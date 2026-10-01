"""
Score every (model, template_id) combo in a sweep_judges.py results file
against the gold set, and print a leaderboard. Pure stdlib -- no
pandas/numpy/torch -- so this runs fine on the login node, no GPU/venv
needed, straight after a sweep job's results land.

Metrics per combo (over the full gold set unless noted):
  - accuracy: judge_reparsed_label is a member of the row's gold_labels set
    (almost always one value; for the ~12 dual-tag "refusal with reasons"
    rows, either valid label counts -- see build_gold_set.py's docstring).
  - accuracy_ci95: cluster-bootstrap 95% CI for accuracy (resampled by
    `stance`, i.e. by statement, not by row -- rows from the same statement
    aren't independent draws).
  - collapsed_accuracy: same idea, but judge/gold collapsed to
    {pro, neutral, con, refusal} buckets first (1,2->pro; 3->neutral;
    4,5->con) -- how often the judge gets the *direction* right even if it
    misses the 1-vs-2 / 4-vs-5 intensity call.
  - unparseable_rate: judge output didn't match a recognized label.

Refusal detection and stance scoring are two different tasks with two
different label spaces (classify_stance.py itself records them in separate
columns, stance_score being NaN for refusal), and this file scores them
separately rather than as one 6-way {1,2,3,4,5,refusal} classification:

  - refusal task (binary: is this row a refusal, judged over ALL rows):
    refusal_accuracy/refusal_precision/refusal_recall/refusal_f1 (of the
    "refusal" class) plus a 2x2 confusion matrix.
  - stance task (5-way ordinal 1-5, judged only over rows whose gold
    includes a directional label -- i.e. excluding rows whose ONLY gold tag
    is a bare, unjustified "refusal"; the ~12 dual-label "refusal with
    reasons" rows DO count here, via their directional tag):
    stance_accuracy (= stance_micro_f1)/stance_macro_f1, qwk (quadratic
    weighted kappa, same subset but additionally requires the judge's
    prediction to itself be directional -- kappa needs a numeric distance,
    so a "refusal"/unparseable prediction can't sit on the 1-5 scale; it
    still counts against stance_accuracy as a miss, just not against qwk)
    plus a 5x6 confusion matrix (columns 1-5 + "other", which catches both
    "refusal" and unparseable predictions).
  - accuracy/accuracy_ci95/collapsed_accuracy stay as a single combined
    headline number across all 500 rows (unaffected by the split above).

For the ~12 dual-label rows, both confusion matrices need one "true" class
per cell (unlike the membership-based `accuracy`): if the judge's own
prediction is one of the row's valid labels, that prediction IS the true
class (so it lands on the diagonal); otherwise the row's directional label
is used over "refusal" as the true class.

Confusion matrices are printed only for the top --top-confusion combos by
default, to avoid flooding stdout on a big sweep; use --all-confusion or
--confusion-for to see more.

results_csv accepts a comma-separated list of paths (e.g. one per model, if
they were swept as separate concurrent jobs -- see sweep_judges.py's
CAUTION about concurrent writers) and concatenates them.

Usage:
    python score_judges.py sweep_results_a.csv,sweep_results_b.csv gold_stance_500.csv [--sort accuracy]
        [--top-confusion 3] [--all-confusion] [--confusion-for MODEL:TEMPLATE_ID ...]
"""

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict

csv.field_size_limit(sys.maxsize)

STANCE_CLASSES = ("1", "2", "3", "4", "5")
REFUSAL_CLASSES = ("refusal", "other")
BUCKET = {"1": "pro", "2": "pro", "3": "neutral", "4": "con", "5": "con", "refusal": "refusal"}
N_BOOTSTRAP = 2000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("results_csv")
    p.add_argument("gold_csv")
    p.add_argument("--sort", default="accuracy", choices=("accuracy", "collapsed_accuracy", "qwk", "refusal_f1", "stance_accuracy", "stance_macro_f1"))
    p.add_argument("--output-csv", default=None, help="optional: also write the scalar leaderboard as CSV")
    p.add_argument("--output-json", default=None, help="optional: write full per-combo detail (scalars + both confusion matrices + per-class breakdowns) as JSON, for downstream analysis/plotting")
    p.add_argument("--top-confusion", type=int, default=3, help="print full confusion matrix + per-label report for the top N combos by --sort")
    p.add_argument("--all-confusion", action="store_true", help="print confusion matrices for every combo, not just the top N")
    p.add_argument("--confusion-for", action="append", default=[], metavar="MODEL:TEMPLATE_ID", help="also force-print this combo's confusion matrix; repeatable")
    return p.parse_args()


def quadratic_weighted_kappa(pairs):
    """pairs: list of (gold_int, pred_int) in 1..5. Pure-python QWK."""
    n_classes = 5
    counts = Counter(pairs)
    gold_marg = Counter(g for g, _ in pairs)
    pred_marg = Counter(p for _, p in pairs)
    n = len(pairs)
    if n == 0:
        return None

    weights = [[((i - j) ** 2) / ((n_classes - 1) ** 2) for j in range(1, n_classes + 1)] for i in range(1, n_classes + 1)]

    observed = sum(weights[g - 1][p - 1] * c for (g, p), c in counts.items())
    expected = sum(
        weights[g - 1][p - 1] * gold_marg.get(g, 0) * pred_marg.get(p, 0) / n
        for g in range(1, n_classes + 1) for p in range(1, n_classes + 1)
    )
    if expected == 0:
        return None
    return 1 - observed / expected


def true_label(r):
    """Single "true" class for a row, for confusion-matrix purposes -- see
    module docstring for the dual-label tiebreak rule."""
    pred = r["judge_reparsed_label"]
    gold_set = r["gold_labels"]
    if pred is not None and pred in gold_set:
        return pred
    directional = [g for g in gold_set if g != "refusal"]
    return directional[0] if directional else next(iter(gold_set))


def build_refusal_confusion(rows):
    """2x2 matrix over {"refusal", "other"}, judged across ALL rows."""
    matrix = Counter()
    for r in rows:
        true = "refusal" if true_label(r) == "refusal" else "other"
        pred = "refusal" if r["judge_reparsed_label"] == "refusal" else "other"
        matrix[(true, pred)] += 1
    return matrix


def build_stance_confusion(rows):
    """5x6 matrix, rows/true in STANCE_CLASSES, cols/pred in STANCE_CLASSES +
    "other" (catches "refusal" and unparseable predictions alike). Only
    counts rows whose gold includes a directional label -- a row whose ONLY
    gold tag is a bare "refusal" isn't part of the stance task at all.
    Also returns qwk_pairs: the (true, pred) subset restricted to predictions
    that are themselves directional, since kappa needs a numeric distance.
    """
    matrix = Counter()
    qwk_pairs = []
    for r in rows:
        if not any(g != "refusal" for g in r["gold_labels"]):
            continue  # refusal-only gold row: stance task doesn't apply
        true = true_label(r)
        pred = r["judge_reparsed_label"]
        pred_bucket = pred if pred in STANCE_CLASSES else "other"
        matrix[(true, pred_bucket)] += 1
        if pred in STANCE_CLASSES:
            qwk_pairs.append((int(true), int(pred)))
    return matrix, qwk_pairs


def prf_from_confusion(matrix, classes):
    """-> (per_class dict, micro_f1, macro_f1). zero_division -> 0.0, sklearn-style."""
    per_class = {}
    for c in classes:
        tp = matrix.get((c, c), 0)
        fp = sum(v for (t, p), v in matrix.items() if p == c and t != c)
        fn = sum(v for (t, p), v in matrix.items() if t == c and p != c)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}

    n = sum(matrix.values())
    micro_f1 = sum(matrix.get((c, c), 0) for c in classes) / n if n else None
    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(classes)
    return per_class, micro_f1, macro_f1


def format_confusion(title, matrix, per_class, true_classes, pred_cols):
    cellw = 8
    lines = [f"\n--- {title} (rows=true, cols=predicted) ---"]
    lines.append("".ljust(cellw) + "".join(c.ljust(cellw) for c in pred_cols))
    for true_c in true_classes:
        row = [str(matrix.get((true_c, p), 0)).ljust(cellw) for p in pred_cols]
        lines.append(true_c.ljust(cellw) + "".join(row))
    lines.append(f"\n{'label':10}{'precision':10}{'recall':10}{'f1':10}{'support':10}")
    for c in true_classes:
        v = per_class[c]
        lines.append(f"{c:10}{v['precision']:<10.3f}{v['recall']:<10.3f}{v['f1']:<10.3f}{v['support']:<10}")
    return "\n".join(lines)


def confusion_to_dict(matrix, true_classes, pred_cols):
    """Counter with (true, pred) tuple keys -> {true: {pred: count}}, since
    tuple keys aren't JSON-serializable."""
    return {t: {p: matrix.get((t, p), 0) for p in pred_cols} for t in true_classes}


def score_rows(rows):
    """rows: list of dicts with gold_labels (set of str) and judge_reparsed_label (str or None)."""
    n = len(rows)
    correct = sum(1 for r in rows if r["judge_reparsed_label"] in r["gold_labels"])
    accuracy = correct / n

    collapsed_correct = 0
    for r in rows:
        jb = BUCKET.get(r["judge_reparsed_label"])
        if jb is not None and jb in {BUCKET[g] for g in r["gold_labels"]}:
            collapsed_correct += 1
    collapsed_accuracy = collapsed_correct / n

    n_unparseable = sum(1 for r in rows if not r["judge_reparsed_label"])

    refusal_confusion = build_refusal_confusion(rows)
    refusal_per_class, refusal_accuracy, _ = prf_from_confusion(refusal_confusion, REFUSAL_CLASSES)

    stance_confusion, qwk_pairs = build_stance_confusion(rows)
    stance_per_class, stance_accuracy, stance_macro_f1 = prf_from_confusion(stance_confusion, STANCE_CLASSES)
    qwk = quadratic_weighted_kappa(qwk_pairs)
    stance_n = sum(stance_confusion.values())

    return {
        "n": n, "accuracy": accuracy, "collapsed_accuracy": collapsed_accuracy,
        "unparseable_rate": n_unparseable / n,
        "refusal_accuracy": refusal_accuracy,
        "refusal_precision": refusal_per_class["refusal"]["precision"],
        "refusal_recall": refusal_per_class["refusal"]["recall"],
        "refusal_f1": refusal_per_class["refusal"]["f1"],
        "refusal_confusion": refusal_confusion, "refusal_per_class": refusal_per_class,
        "stance_n": stance_n, "stance_accuracy": stance_accuracy, "stance_macro_f1": stance_macro_f1,
        "qwk": qwk, "qwk_n": len(qwk_pairs),
        "stance_confusion": stance_confusion, "stance_per_class": stance_per_class,
    }


def bootstrap_accuracy_ci(rows, by_key):
    """Cluster bootstrap over by_key (e.g. stance statement), N_BOOTSTRAP reps."""
    clusters = defaultdict(list)
    for r in rows:
        clusters[by_key(r)].append(r)
    keys = list(clusters)
    accs = []
    for _ in range(N_BOOTSTRAP):
        resampled = [row for k in random.choices(keys, k=len(keys)) for row in clusters[k]]
        correct = sum(1 for r in resampled if r["judge_reparsed_label"] in r["gold_labels"])
        accs.append(correct / len(resampled))
    accs.sort()
    lo = accs[int(0.025 * N_BOOTSTRAP)]
    hi = accs[int(0.975 * N_BOOTSTRAP)]
    return lo, hi


def main():
    args = parse_args()
    random.seed(0)

    gold_by_row = {r["row_number"]: r for r in csv.DictReader(open(args.gold_csv, newline=""))}

    grouped = defaultdict(list)
    for path in args.results_csv.split(","):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                gold_row = gold_by_row[r["row_number"]]
                grouped[(r["model"], r["template_id"])].append({
                    "gold_labels": set(r["gold_labels"].split(";")),
                    "judge_reparsed_label": r["judge_reparsed_label"] or None,
                    "stance": gold_row["stance"],
                })

    leaderboard = []
    for (model, template_id), rows in grouped.items():
        m = score_rows(rows)
        lo, hi = bootstrap_accuracy_ci(rows, by_key=lambda r: r["stance"])
        m["accuracy_ci95"] = f"[{lo:.3f}, {hi:.3f}]"
        leaderboard.append({"model": model, "template_id": template_id, **m})

    leaderboard.sort(key=lambda m: (m[args.sort] if m[args.sort] is not None else -1), reverse=True)

    header = ["model", "template_id", "n", "accuracy", "accuracy_ci95", "collapsed_accuracy",
              "refusal_accuracy", "refusal_precision", "refusal_recall", "refusal_f1",
              "stance_n", "stance_accuracy", "stance_macro_f1", "qwk", "qwk_n", "unparseable_rate"]
    widths = {"model": 28, "template_id": 10, "n": 4, "accuracy": 8, "accuracy_ci95": 16,
              "collapsed_accuracy": 10, "refusal_accuracy": 9, "refusal_precision": 9,
              "refusal_recall": 9, "refusal_f1": 9, "stance_n": 9, "stance_accuracy": 9,
              "stance_macro_f1": 9, "qwk": 6, "qwk_n": 5, "unparseable_rate": 8}

    def fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    print(" ".join(h.ljust(widths[h]) for h in header))
    for m in leaderboard:
        print(" ".join(fmt(m[h]).ljust(widths[h]) for h in header))

    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(leaderboard)
        print(f"\nWrote leaderboard to {args.output_csv}")

    if args.output_json:
        detail = [
            {
                **{h: m[h] for h in header},
                "refusal_confusion": confusion_to_dict(m["refusal_confusion"], REFUSAL_CLASSES, REFUSAL_CLASSES),
                "refusal_per_class": m["refusal_per_class"],
                "stance_confusion": confusion_to_dict(m["stance_confusion"], STANCE_CLASSES, list(STANCE_CLASSES) + ["other"]),
                "stance_per_class": m["stance_per_class"],
            }
            for m in leaderboard
        ]
        with open(args.output_json, "w") as f:
            json.dump(detail, f, indent=2)
        print(f"Wrote full per-combo detail to {args.output_json}")

    forced = set(args.confusion_for)
    for rank, m in enumerate(leaderboard):
        combo_key = f"{m['model']}:{m['template_id']}"
        if args.all_confusion or rank < args.top_confusion or combo_key in forced:
            title = f"{m['model']} / {m['template_id']}"
            print(format_confusion(f"refusal task: {title}", m["refusal_confusion"], m["refusal_per_class"],
                                    REFUSAL_CLASSES, list(REFUSAL_CLASSES)))
            print(format_confusion(f"stance task: {title}", m["stance_confusion"], m["stance_per_class"],
                                    STANCE_CLASSES, list(STANCE_CLASSES) + ["other"]))


if __name__ == "__main__":
    main()
