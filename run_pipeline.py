"""
Convenience orchestrator: run generate_texts.py, then (if configured)
classify_stance.py and classify_refusal.py, all against one pipeline
config, for one model.

Each stage runs as its own subprocess, so its GPU memory is fully released
when it exits before the next stage starts -- generation and stance
classification load a 7-8B causal LM, while refusal classification loads a
much smaller encoder, and holding all of them at once in one process would
size the job to the biggest stage for no reason.

This is meant for the common case of running everything on one interactive
/ single-allocation GPU. On a job-scheduled cluster you can just as well
submit generate_texts.py / classify_stance.py / classify_refusal.py as
separate jobs (each sized to what that stage needs) pointed at the same
config, chained via job dependencies -- run_pipeline.py is optional.

Usage:
    python run_pipeline.py config.yml
"""

import subprocess
import sys
from pathlib import Path

from pipeline_config import load_config

HERE = Path(__file__).resolve().parent


def run_stage(script_name, config_path):
    print(f"\n=== {script_name} ===")
    subprocess.run([sys.executable, str(HERE / script_name), config_path], check=True)


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: python {sys.argv[0]} config.yml")
    config_path = sys.argv[1]
    cfg = load_config(config_path)

    run_stage("generate_texts.py", config_path)

    if "stance_classifier" in cfg:
        run_stage("classify_stance.py", config_path)
    else:
        print("\n(no stance_classifier section -- skipping stance classification)")

    if "refusal_classifier" in cfg:
        run_stage("classify_refusal.py", config_path)
    else:
        print("\n(no refusal_classifier section -- skipping refusal classification)")

    print(f"\nDone. Outputs in {cfg['output_dir']}")


if __name__ == "__main__":
    main()
