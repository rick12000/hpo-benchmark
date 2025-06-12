import ast
from hpobench.config.config import IntRange, CategoricalRange, FloatRange
import random
from typing import Optional, Union
import pandas as pd
import os
import logging
from hpobench.generation.generate import ObjectiveMetricGenerator

logger = logging.getLogger(__name__)


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
    output_folder: str = "data",
):
    if df is not None and not df.empty:
        analysis_data_path = os.path.join(cache_path, output_folder, run_start_str)
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
    """Adds cumulative generator runtime and total runtime to the experiment log."""
    experiment_log_copy = experiment_log.copy()

    # Calculate cumulative runtime for the generator predictions
    experiment_log_copy["generator_runtime"] = experiment_log_copy[
        "configurations"
    ].apply(lambda x: performance_generator.predict_runtime(x))
    experiment_log_copy["generator_runtime"] = experiment_log_copy[
        "generator_runtime"
    ].cumsum()

    # Calculate total runtime (tuner time + generator time)
    experiment_log_copy["runtime"] = (
        experiment_log_copy["end_time"] - tune_start
    ).dt.total_seconds()  # Use total_seconds() for float representation
    experiment_log_copy["runtime"] = (
        experiment_log_copy["runtime"] + experiment_log_copy["generator_runtime"]
    )

    return experiment_log_copy
