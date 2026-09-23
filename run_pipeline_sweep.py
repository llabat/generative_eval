"""
Run run_pipeline.py once per model, sequentially (one model at a time, so
only one model's worth of GPU memory is ever in use), from a single base
config plus a list of models.

Base config: everything a normal pipeline config has (see
pipeline_config.py's docstring) EXCEPT `model:`, plus:

    models:
      - /path/to/model_a
      - /path/to/model_b
    output_dir: runs/          # base dir; each model gets runs/<model_name>/

For each model, a per-model config is derived (`model:` set to that path,
`output_dir:` set to <base output_dir>/<model_dir_name>) and written to
<per-model output_dir>/config.yml, then run_pipeline.py is invoked on it.

Usage:
    python run_pipeline_sweep.py sweep_config.yml
"""

import os
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def load_sweep_config(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    for key in ("models", "output_dir"):
        if key not in cfg:
            raise ValueError(f"{config_path} is missing required key: {key}")
    if "model" in cfg:
        raise ValueError(f"{config_path} should list models under `models:`, not a single `model:`")

    return cfg


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: python {sys.argv[0]} sweep_config.yml")
    sweep_cfg = load_sweep_config(sys.argv[1])

    models = sweep_cfg.pop("models")
    base_output_dir = sweep_cfg.pop("output_dir")

    for model_path in models:
        model_name = model_path.rstrip("/").split("/")[-1]
        per_model_cfg = dict(sweep_cfg)
        per_model_cfg["model"] = model_path
        per_model_cfg["output_dir"] = os.path.join(base_output_dir, model_name)

        os.makedirs(per_model_cfg["output_dir"], exist_ok=True)
        config_path = os.path.join(per_model_cfg["output_dir"], "config.yml")
        with open(config_path, "w") as f:
            yaml.safe_dump(per_model_cfg, f, sort_keys=False)

        print(f"\n##### {model_name} #####")
        subprocess.run([sys.executable, str(HERE / "run_pipeline.py"), config_path], check=True)

    print(f"\nSwept {len(models)} models, outputs under {base_output_dir}")


if __name__ == "__main__":
    main()
