"""
One-time export of the MEVS opinion set (mevs_opinions.py's ID2OPINION,
which reads from a hardcoded MEVS.json path in the polyglots repo) to a
plain `question_id, stance` CSV.

The pipeline itself (prompts.py / generate_texts.py) never imports
mevs_opinions.py -- it only reads the CSV this script produces, so it works
with any opinion set, not just MEVS. Use the resulting CSV as
`prompts.opinions_csv` in a pipeline config, paired with `prompts.templates_csv`.

Usage:
    python export_mevs_opinions.py
    python export_mevs_opinions.py --output-csv mevs_opinions.csv
"""

import argparse

import pandas as pd

from mevs_opinions import ID2OPINION


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", default="mevs_opinions.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.DataFrame(
        {"question_id": list(ID2OPINION.keys()), "stance": list(ID2OPINION.values())}
    )
    df.to_csv(args.output_csv, index=False)
    print(f"Saved {len(df)} opinions to {args.output_csv}")


if __name__ == "__main__":
    main()
