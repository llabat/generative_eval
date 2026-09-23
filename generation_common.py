"""
Backend-agnostic pieces shared by generate_texts.py's transformers backend
and vllm_backend.py: the resume/CSV bookkeeping (identical regardless of
which backend produced the rows) and the generation_config.json fallback
used to resolve a sampling config with empty `params`.
"""

import json
import os

import pandas as pd

# Some of our SFT checkpoints don't ship their own sampling params in
# generation_config.json (only eos/pad/bos ids). Rather than silently
# falling back to transformers' bare library defaults (temperature=1.0,
# top_p=1.0) for those, we apply this explicit, shared fallback whenever a
# model has no temperature/top_p of its own and its sampling config's
# `params` is empty ({}).
FALLBACK_SAMPLING = {"temperature": 0.7, "top_p": 0.9}

GROUP_KEYS = ["prompt_id", "model", "config"]


def model_own_sampling_overrides(model_path):
    """Read a model's raw generation_config.json (if any) and decide what
    override to apply for a sampling config with empty `params`: {} if the
    model ships its own temperature/top_p (use what's attached to the model
    object), else FALLBACK_SAMPLING.
    """
    config_path = os.path.join(model_path, "generation_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            raw_config = json.load(f)
        if any(k in raw_config for k in ("temperature", "top_p")):
            return {}
    return dict(FALLBACK_SAMPLING)


def resolve_sampling_kwargs(params, model_default_overrides):
    return dict(model_default_overrides) if not params else dict(params)


def fieldnames_for(df):
    """prompt_id, prompt, then any extra metadata columns from the prompt
    set (question_id/stance/template_id/... from prompts.py, or whatever a
    dataset_csv carries), then the columns generation itself adds.
    """
    metadata_cols = [c for c in df.columns if c not in ("prompt_id", "prompt")]
    return ["prompt_id", "prompt"] + metadata_cols + ["model", "config", "generation_id", "response"]


def load_completed_pairs(output_csv, num_generations_by_config):
    """Inspect an existing output CSV and figure out which (prompt_id, model,
    config) batches are already fully generated -- "fully" meaning at least
    that config's own num_generations -- so a resumed run can skip them.

    Returns (completed, kept_df):
      - completed: set of (prompt_id, model, config) with enough rows.
      - kept_df: the rows for those completed batches, trimmed to exactly
        num_generations each (None if there's nothing to keep).
    """
    if not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0:
        return set(), None

    try:
        existing = pd.read_csv(output_csv)
    except pd.errors.EmptyDataError:
        return set(), None

    if existing.empty or "config" not in existing.columns:
        # Either genuinely empty, or an incompatible schema -- don't try to
        # resume against it.
        return set(), None

    counts = existing.groupby(GROUP_KEYS).size()
    needed = counts.index.get_level_values("config").map(num_generations_by_config)
    completed = set(counts[counts >= needed].index)

    if not completed:
        return set(), None

    is_complete = existing.set_index(GROUP_KEYS).index.isin(completed)
    complete_rows = existing[is_complete]
    # Trim each complete group down to exactly its own num_generations,
    # without relying on groupby(...).apply() -- newer pandas deprecates
    # passing the grouping columns (prompt_id/model/config) through to the
    # applied function, which silently dropped them from the result here.
    rank_in_group = complete_rows.groupby(GROUP_KEYS).cumcount()
    needed_per_row = complete_rows["config"].map(num_generations_by_config)
    kept = complete_rows[rank_in_group < needed_per_row]
    return completed, kept
