"""
One-off: turn the hand-annotated annotations_generations-stance_all_train.csv
into a clean gold set for scoring stance judges against.

Input quirks this untangles:
  - Only the `stance-refusal_labels` column is ever populated (the plain
    `stance_labels` column is empty for all rows) -- so that's the one true
    label column, despite its name suggesting it's refusal-specific.
  - `text` is "PROMPT:\n<generation prompt>\n\nRESPONSE:\n<model response>",
    and the generation prompt embeds the bare stance statement inside
    whichever of the ~60 generation templates (templates.csv) was used to
    phrase it. Rather than trying to reverse each template's phrasing, this
    recovers the bare statement by matching against the closed vocabulary of
    known statements in mevs_opinions.csv -- every labeled row's prompt
    contains exactly one of those verbatim (verified for all 500 rows).
  - `stance-refusal_labels` is a "|"-joined bag of tags: a compliance
    category (NORMAL / DISCLAIMER_TECHNICAL / DISCLAIMER_ETHICAL /
    REFUSAL_TECHNICAL / REFUSAL_ETHICAL) plus one or more lowercase stance
    tags (only_pro/mostly_pro/neutral/mostly_con/only_con/refusal). Per the
    annotator: a bare "refusal" tag means a formulaic refusal with no
    justification; "refusal" alongside a directional tag (e.g.
    "refusal|only_con") means a refusal that still argues a direction.
    For those ~13 dual-tag rows there's no single correct judge answer, so
    gold_labels holds BOTH valid labels (";"-joined) -- a judge is scored
    correct if it outputs either one (decided with the user; see
    score_judges.py).

Output columns: row_number, stance, response, category, raw_label,
gold_labels (";"-joined subset of {1,2,3,4,5,refusal} -- almost always one
value, ";"-joined only for the dual-tag rows).

Usage:
    python build_gold_set.py annotations_generations-stance_all_train.csv gold_stance_500.csv
"""

import csv
import re
import sys

STANCE_TO_LABEL = {
    "only_pro": "1",
    "mostly_pro": "2",
    "neutral": "3",
    "mostly_con": "4",
    "only_con": "5",
    "refusal": "refusal",
}
PROMPT_RESPONSE_RE = re.compile(r"PROMPT:\n(.*?)\n\nRESPONSE:\n(.*)", re.S)


def split_prompt_response(text):
    m = PROMPT_RESPONSE_RE.match(text)
    if not m:
        raise ValueError(f"couldn't split PROMPT/RESPONSE out of: {text[:200]!r}")
    return m.group(1), m.group(2)


def recover_stance(prompt, known_stances):
    hits = [s for s in known_stances if s in prompt]
    if len(hits) != 1:
        raise ValueError(f"expected exactly 1 known stance in prompt, found {len(hits)}: {prompt[:200]!r}")
    return hits[0]


def parse_gold_labels(raw_label):
    """-> (category, gold_labels list). category is the one ALL_CAPS tag;
    gold_labels are STANCE_TO_LABEL-mapped values for every lowercase tag."""
    tags = raw_label.split("|")
    category = next((t for t in tags if t.isupper()), None)
    stance_tags = [t for t in tags if not t.isupper()]
    if category is None or not stance_tags:
        raise ValueError(f"couldn't parse category/stance tags out of: {raw_label!r}")
    gold_labels = [STANCE_TO_LABEL[t] for t in stance_tags]
    return category, gold_labels


def main():
    if len(sys.argv) != 3:
        raise SystemExit(f"Usage: python {sys.argv[0]} annotations_csv output_csv")
    annotations_csv, output_csv = sys.argv[1], sys.argv[2]

    known_stances = [r["stance"] for r in csv.DictReader(open("mevs_opinions.csv", newline=""))]

    out_rows = []
    with open(annotations_csv, newline="") as f:
        for r in csv.DictReader(f):
            raw_label = r["stance-refusal_labels"].strip()
            if not raw_label:
                continue
            prompt, response = split_prompt_response(r["text"])
            stance = recover_stance(prompt, known_stances)
            category, gold_labels = parse_gold_labels(raw_label)
            out_rows.append({
                "row_number": r["row_number"],
                "stance": stance,
                "response": response,
                "category": category,
                "raw_label": raw_label,
                "gold_labels": ";".join(gold_labels),
            })

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["row_number", "stance", "response", "category", "raw_label", "gold_labels"])
        writer.writeheader()
        writer.writerows(out_rows)

    n_dual = sum(1 for r in out_rows if ";" in r["gold_labels"])
    print(f"Wrote {len(out_rows)} gold rows to {output_csv} ({n_dual} with dual valid labels)")


if __name__ == "__main__":
    main()
