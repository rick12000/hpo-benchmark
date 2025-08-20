import ast
from hpobench.config.config_types import IntRange, CategoricalRange, FloatRange
import random
from typing import Optional, Union, List
import pandas as pd
import os
import logging
from hpobench.generation.generate import ObjectiveMetricGenerator
from datetime import datetime
import optuna
import numpy as np

logger = logging.getLogger(__name__)


class AnalysisPathManager:
    """Simple path manager for organizing analysis outputs by type and purpose."""

    def __init__(self, cache_path: str, run_start_str: str):
        self.cache_path = cache_path
        self.run_start_str = run_start_str
        self.base_path = os.path.join(cache_path, "experiments", run_start_str)

    def get_analysis_path(
        self, analysis_type: str, output_type: str = "data", subfolder: str = None
    ) -> str:
        """Get path for specific analysis type and output type.

        Args:
            analysis_type: e.g., "01_coverage_analysis", "02_sampler_variation"
            output_type: "data" or "plots"
            subfolder: optional subfolder like "statistical_tests", "aggregated_results"
        """
        path = os.path.join(self.base_path, analysis_type, output_type)
        if subfolder:
            path = os.path.join(path, subfolder)
        os.makedirs(path, exist_ok=True)
        return path


def q10(x):
    return x.quantile(0.1)


def q90(x):
    return x.quantile(0.9)


def get_group_dict(breakout_col, within_group):
    if breakout_col is None:
        return {}
    if isinstance(within_group, tuple):
        return dict(zip(breakout_col, within_group))
    return {breakout_col[0]: within_group}


def generate_hyperparameter_combinations(
    params: dict[str, Union[IntRange, FloatRange, CategoricalRange]],
    n_combinations: int,
    random_state: Optional[int] = None,
):
    random.seed(random_state)
    combinations = []
    for _ in range(n_combinations):
        combination = {}
        for param_name, param_values in params.items():
            if isinstance(param_values, IntRange):
                combination[param_name] = random.choice(
                    list(range(param_values.lower, param_values.upper + 1))
                )
            elif isinstance(param_values, FloatRange):
                combination[param_name] = random.choice(
                    [
                        random.uniform(param_values.lower, param_values.upper)
                        for _ in range(1000)
                    ]
                )
            elif isinstance(param_values, CategoricalRange):
                combination[param_name] = random.choice(param_values.choices)
            else:
                raise ValueError()
        combinations.append(combination)
    return combinations


def parse_config_space(s, openml_id: str):
    config_dict = {}
    for line in s.split("\n"):
        line = line.strip()
        if not line or line in {"Configuration space object:", "Hyperparameters:"}:
            continue

        # Custom parser to handle commas inside brackets/braces
        parts = []
        current = []
        in_bracket = False
        bracket_chars = {"[", "{"}

        for char in line:
            if char in bracket_chars:
                in_bracket = True
            elif char in {"]", "}"}:
                in_bracket = False

            if char == "," and not in_bracket:
                parts.append("".join(current).strip())
                current = []
            else:
                current.append(char)
        if current:
            parts.append("".join(current).strip())

        if not parts:
            continue

        name = parts[0]
        param_type = None
        choices = None
        range_values = None
        value = None

        for part in parts[1:]:
            if part.startswith("Type: "):
                param_type = part.split(": ")[1]
            elif part.startswith("Choices: "):
                choices_str = part.split(": ")[1].strip()
                # Convert curly braces to list format
                if choices_str.startswith("{"):
                    choices_str = f"[{choices_str[1:-1]}]"
                choices = ast.literal_eval(choices_str) if choices_str else []
            elif part.startswith("Range: "):
                range_str = part.split(": ")[1].strip("[]")
                range_values = (
                    [x.strip() for x in range_str.split(",")] if range_str else []
                )
            elif part.startswith("Value: "):
                value = (
                    ast.literal_eval(part.split(": ")[1])
                    if part.split(": ")[1]
                    else None
                )

        # Handle parameter types
        if param_type == "Categorical" and choices is not None:
            config_dict[name] = CategoricalRange(choices=choices)
        elif param_type == "UniformInteger" and range_values:
            values = [int(x) for x in range_values]
            config_dict[name] = IntRange(lower=values[0], upper=values[1])
        elif param_type == "UniformFloat" and range_values:
            values = [float(x) for x in range_values]
            config_dict[name] = FloatRange(lower=values[0], upper=values[1])
        elif param_type == "Constant" and value is not None:
            config_dict[name] = CategoricalRange(choices=[value])

    config_dict["OpenML_task_id"] = CategoricalRange(choices=[openml_id])

    return config_dict


def save_analysis_results(
    df: pd.DataFrame,
    cache_path: str,
    run_start_str: str,
    filename: str,
    analysis_type: str = None,
    subfolder: str = None,
):
    """Save analysis results with proper path organization.

    Args:
        df: DataFrame to save
        cache_path: Base cache path
        run_start_str: Run identifier
        filename: Name of the file
        analysis_type: Analysis type (e.g., "01_coverage_analysis")
        subfolder: Optional subfolder (e.g., "statistical_tests")
    """
    if df is not None and not df.empty:
        if analysis_type:
            path_manager = AnalysisPathManager(cache_path, run_start_str)
            analysis_data_path = path_manager.get_analysis_path(
                analysis_type, "data", subfolder
            )
        else:
            # Fallback to old behavior for backward compatibility
            analysis_data_path = os.path.join(cache_path, "data", run_start_str)
            os.makedirs(analysis_data_path, exist_ok=True)

        full_filename = os.path.join(analysis_data_path, filename)
        try:
            df.to_csv(full_filename, index=False)
            logger.info(f"Saved results to {full_filename}")
        except Exception as e:
            logger.error(
                f"Failed to save results to {full_filename}: {e}", exc_info=True
            )
    else:
        logger.warning(f"Skipping save for {filename}: DataFrame is empty or None.")


def add_runtime(
    experiment_log: pd.DataFrame,
    tune_start,
    performance_generator: ObjectiveMetricGenerator,
):
    experiment_log_copy = experiment_log.copy()

    experiment_log_copy["generator_runtime"] = experiment_log_copy[
        "configurations"
    ].apply(lambda x: performance_generator.predict_runtime(x))
    experiment_log_copy["generator_runtime"] = experiment_log_copy[
        "generator_runtime"
    ].cumsum()

    experiment_log_copy["runtime"] = (
        experiment_log_copy["end_time"] - tune_start
    ).dt.total_seconds()
    experiment_log_copy["runtime"] = (
        experiment_log_copy["runtime"] + experiment_log_copy["generator_runtime"]
    )

    return experiment_log_copy


def setup_environment(cache_path: str = "cache/") -> tuple[str, logging.Logger]:
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)

    run_start = datetime.now()
    run_start_str = run_start.strftime("%Y-%m-%d_%H-%M-%S")

    log_path = os.path.join(cache_path, f"logs/{run_start_str}")
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_filename = os.path.join(
        log_path, f"run_{run_start.strftime(format='%m_%d_%Y-%H_%M_%S')}.log"
    )
    logging.basicConfig(
        filename=log_filename,
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logger = logging.getLogger()

    logging.getLogger("hyperopt").setLevel(logging.ERROR)
    logging.getLogger("confopt").setLevel(logging.ERROR)
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    logging.getLogger("yahpo").setLevel(logging.WARNING)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return run_start_str, logger


def block_bootstrap(
    data: pd.DataFrame,
    breakout_cols: List[str],
    block_cols: List[str],
    aggregators: List[str],
    metric_cols: List[str],
    n_bootstraps: int,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Compute block-bootstrap percentile intervals for grouped metrics.

    Performs a block bootstrap within each combination of `breakout_cols`.
    Blocks are defined by `block_cols` and are resampled with replacement to
    create bootstrap samples. For each bootstrap iteration the mean of the
    `metric_cols` is computed per `aggregators` group. The function returns
    the original group means plus 5th and 95th percentile bounds computed
    over the bootstrap iterations (named "<metric>_lower" and "<metric>_upper").

    Args:
        data: Input DataFrame containing metrics and grouping columns.
        breakout_cols: Columns defining independent breakout groups; bootstrapping
            is performed separately within each breakout group.
        block_cols: Columns whose combined values define a block/key that is
            resampled (preserves within-block dependencies).
        aggregators: Columns to group by when computing means (e.g. experimental
            factors to aggregate over).
        metric_cols: Numeric metric columns to aggregate and compute percentiles for.
        n_bootstraps: Number of bootstrap iterations to perform per breakout group.
        random_state: Optional seed for numpy's RNG to make results reproducible.

    Returns:
        DataFrame: A table containing the original means for each `aggregators`
        group and, for each metric in `metric_cols`, the lower (5th pct) and
        upper (95th pct) bootstrap bounds named "<metric>_lower" and
        "<metric>_upper". If no valid bootstrap could be performed for a
        group, the bounds are NaN.
    """

    if random_state is not None:
        np.random.seed(random_state)

    # Create unique key column from nesting_cols (work with copy for key creation only)
    all_cols = list(set(block_cols + aggregators + metric_cols + breakout_cols))
    data_with_key = data[all_cols].copy()
    data_with_key["_bootstrap_key"] = (
        data_with_key[block_cols].astype(str).agg("_".join, axis=1)
    )

    # Calculate original means for each aggregator group.
    # Keep the original metric column names for the mean aggregation (no "_mean" suffix).
    original_means = data_with_key.groupby(aggregators, as_index=False)[
        metric_cols
    ].mean()

    all_bootstrap_results = []

    # Group by all breakout columns
    for breakout_values, breakout_group in data_with_key.groupby(breakout_cols):
        unique_keys = breakout_group["_bootstrap_key"].unique()
        if len(unique_keys) < 2:
            continue

        bootstrap_iterations = []
        for _ in range(n_bootstraps):
            sampled_keys = np.random.choice(
                unique_keys, size=len(unique_keys), replace=True
            )
            bootstrap_sample = pd.concat(
                [
                    breakout_group[breakout_group["_bootstrap_key"] == key]
                    for key in sampled_keys
                ],
                ignore_index=True,
            )
            bootstrap_means = bootstrap_sample.groupby(aggregators, as_index=False)[
                metric_cols
            ].mean()
            bootstrap_iterations.append(bootstrap_means)

        if bootstrap_iterations:
            combined_bootstrap = pd.concat(bootstrap_iterations, ignore_index=True)
            # Add breakout columns to keep track of group
            if isinstance(breakout_values, tuple):
                for col, val in zip(breakout_cols, breakout_values):
                    combined_bootstrap[col] = val
            else:
                combined_bootstrap[breakout_cols[0]] = breakout_values
            all_bootstrap_results.append(combined_bootstrap)

    if all_bootstrap_results:
        all_bootstrap_data = pd.concat(all_bootstrap_results, ignore_index=True)

        def p5(x):
            return np.percentile(x, 5)

        def p95(x):
            return np.percentile(x, 95)

        agg_dict = {col: [p5, p95] for col in metric_cols}
        group_cols = list(set(breakout_cols + aggregators))
        percentile_results = all_bootstrap_data.groupby(group_cols, as_index=False).agg(
            agg_dict
        )

        new_cols = []
        for col in percentile_results.columns:
            if isinstance(col, tuple):
                if col[1] == "p5":
                    new_cols.append(f"{col[0]}_lower")
                elif col[1] == "p95":
                    new_cols.append(f"{col[0]}_upper")
                else:
                    new_cols.append("_".join([str(c) for c in col if c]))
            else:
                new_cols.append(col)
        percentile_results.columns = new_cols
    else:
        percentile_results = original_means[aggregators].copy()
        for metric in metric_cols:
            percentile_results[f"{metric}_lower"] = np.nan
            percentile_results[f"{metric}_upper"] = np.nan

    final_results = original_means.merge(
        percentile_results, on=aggregators, how="outer"
    )

    return final_results
