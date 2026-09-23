"""
Shared YAML config loading for the generation/classification pipeline
scripts (generate_texts.py, classify_stance.py, classify_refusal.py,
run_pipeline.py, run_pipeline_sweep.py).

Config shape (see README-style example in run_pipeline.py's docstring):

    model: /path/to/generation/model

    sampling_configs:
      - name: model_default
        num_generations: 100
        params: {}                 # {} -> resolved from the model's own
                                    #  generation_config (falling back to
                                    #  FALLBACK_SAMPLING in generate_texts.py
                                    #  if it doesn't have one)
      - name: low_temp
        num_generations: 50
        params: {temperature: 0.3, top_p: 0.9, top_k: 50}

    generation:
      max_new_tokens: 300
      backend: transformers      # optional, default "transformers"; or "vllm"
                                  #  -- see vllm_backend.py

    prompts:
      dataset_csv: /path/to/prompts.csv
      # -- or, mutually exclusive with dataset_csv --
      # opinions_csv: /path/to/opinions.csv     # question_id, stance columns
      # templates_csv: /path/to/templates.csv   # template_id, template_type,
                                                  #  template columns (template_type
                                                  #  the coarse neutral/support/oppose
                                                  #  category, template a literal
                                                  #  `{stance}`-templated string);
                                                  #  cartesian product of the two
                                                  #  builds the prompt set

    stance_classifier:            # omit this section entirely to skip the stance stage
      model: /path/to/judge/model
      template_id: templ-6        # required -- id into templates_csv; see
                                    #  classify_stance.py's docstring
      # templates_csv: adapted_issuebench_stance_prompt_templates.csv  # optional, default shown
      batch_size: 64
      max_new_tokens: 64          # matches IssueBench's own budget -- see
                                    #  classify_stance.py's docstring
      # temperature: 1.0            # optional, default null -> greedy; set to
                                      #  match IssueBench's own protocol instead
      # backend: vllm               # optional, default "transformers" -- see
                                      #  classify_stance.py's docstring
      # tensor_parallel_size: 2      # vllm backend only -- shard a large judge
                                      #  (e.g. 70B) across multiple GPUs

    refusal_classifier:           # omit this section entirely to skip the refusal stage
      checkpoint: /path/to/refusal/checkpoint
      batch_size: 256
      max_length: 4096

    output_dir: runs/some_model
    resume: true
"""

import os

import yaml

REQUIRED_TOP_LEVEL = ("model", "sampling_configs", "prompts", "output_dir")


def load_config(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    missing = [k for k in REQUIRED_TOP_LEVEL if k not in cfg]
    if missing:
        raise ValueError(f"{config_path} is missing required key(s): {missing}")

    dataset_csv = cfg["prompts"].get("dataset_csv")
    templated = "opinions_csv" in cfg["prompts"] and "templates_csv" in cfg["prompts"]
    if bool(dataset_csv) == bool(templated):
        raise ValueError(
            f"{config_path}: prompts section must set exactly one of "
            "`dataset_csv` or (`opinions_csv` + `templates_csv`)"
        )

    return cfg


def output_path(cfg, filename):
    return os.path.join(cfg["output_dir"], filename)
