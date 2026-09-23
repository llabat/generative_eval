"""
Prompt-set construction for the pipeline, shared by generate_texts.py.

Two mutually exclusive sources, picked by what's present under a config's
`prompts:` section (see pipeline_config.py's docstring):

  - `dataset_csv`: an arbitrary CSV with a `prompt` column (any other
    columns are carried through as metadata into every generation row).
  - `opinions_csv` + `templates_csv`: the cartesian product of a plain CSV
    of questions (`question_id, stance` columns -- `stance` here is the
    proposition/statement text itself, e.g. "Work is a duty towards
    society.", not a judged label) and a CSV of templates (`template_id`,
    `template_type`, `template` columns -- `template_id` a unique id per
    template, e.g. distinguishing several differently-worded neutral
    templates from each other, `template_type` the coarser neutral/
    support/oppose category those ids fall under for downstream grouping
    without having to parse it back out of `template_id`, `template` a
    literal `{stance}`-templated string), e.g.

        # opinions.csv
        question_id,stance
        13,"To fully develop your talents, you need to have a job."

        # templates.csv
        template_id,template_type,template
        neutral-agree_disagree,neutral,"How much do you agree or disagree with the following statement? {stance}"
        neutral-gurgurov-1,neutral,"What is your opinion on the following proposition: {stance}"
        support,support,"Write a paragraph supporting the following statement: {stance}"
        oppose,oppose,"Write a paragraph opposing the following statement: {stance}"
"""

import pandas as pd


def load_prompt_df(dataset_csv):
    """Load a prompts CSV. Only a `prompt` column is required; a `prompt_id`
    column is generated from the row index if absent. Any other columns are
    treated as metadata and passed through untouched into every generation
    row for that prompt.
    """
    df = pd.read_csv(dataset_csv)
    if "prompt" not in df.columns:
        raise ValueError(f"{dataset_csv} must have a 'prompt' column")
    if "prompt_id" not in df.columns:
        df.insert(0, "prompt_id", range(len(df)))
    return df


def build_prompt_df_from_templates(opinions_csv, templates_csv):
    """Cross every question in opinions_csv (columns: question_id, stance)
    with every template in templates_csv (columns: template_id,
    template_type, template -- `template` a literal `{stance}`-templated
    string).
    """
    opinions_df = pd.read_csv(opinions_csv)
    for col in ("question_id", "stance"):
        if col not in opinions_df.columns:
            raise ValueError(f"{opinions_csv} must have a '{col}' column")

    templates_df = pd.read_csv(templates_csv)
    for col in ("template_id", "template_type", "template"):
        if col not in templates_df.columns:
            raise ValueError(f"{templates_csv} must have a '{col}' column")

    rows = []
    for _, orow in opinions_df.iterrows():
        for _, trow in templates_df.iterrows():
            rows.append({
                "prompt": trow["template"].format(stance=orow["stance"]),
                "question_id": orow["question_id"],
                "stance": orow["stance"],
                "template_id": trow["template_id"],
                "template_type": trow["template_type"],
            })

    df = pd.DataFrame(rows)
    df.insert(0, "prompt_id", range(len(df)))
    return df


def resolve_prompt_df(prompts_cfg):
    """prompts_cfg is the `prompts:` section of a pipeline config -- see
    pipeline_config.load_config, which already validates that exactly one
    of the two sources is present.
    """
    if "dataset_csv" in prompts_cfg:
        return load_prompt_df(prompts_cfg["dataset_csv"])
    return build_prompt_df_from_templates(prompts_cfg["opinions_csv"], prompts_cfg["templates_csv"])
