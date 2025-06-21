import ast
from hpobench.config.types import IntRange, CategoricalRange, FloatRange
import random
from typing import Optional, Union
import pandas as pd
import os
import logging
from hpobench.generation.generate import ObjectiveMetricGenerator
from datetime import datetime
import optuna

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
    description: str,
    analysis_type: str = None,
    subfolder: str = None,
):
    """Save analysis results with proper path organization.

    Args:
        df: DataFrame to save
        cache_path: Base cache path
        run_start_str: Run identifier
        filename: Name of the file
        description: Description for logging
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
            logger.info(f"{description} saved to {full_filename}")
        except Exception as e:
            logger.error(
                f"Failed to save {description} to {full_filename}: {e}", exc_info=True
            )
    else:
        logger.warning(f"Skipping save for {description}: DataFrame is empty or None.")


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
