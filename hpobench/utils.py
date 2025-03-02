import ast
from hpobench.config import IntRange, CategoricalRange, FloatRange
import random
from typing import Optional, Union


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
            if param_values.type == "int":
                combination[param_name] = random.choice(
                    list(range(param_values.lower, param_values.upper + 1))
                )
            elif param_values.type == "float":
                combination[param_name] = random.choice(
                    [
                        random.uniform(param_values.lower, param_values.upper)
                        for _ in range(1000)
                    ]
                )
            elif param_values.type == "categorical":
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
                choices = ast.literal_eval(choices_str)
            elif part.startswith("Range: "):
                range_str = part.split(": ")[1].strip("[]")
                range_values = [x.strip() for x in range_str.split(",")]
            elif part.startswith("Value: "):
                value = ast.literal_eval(part.split(": ")[1])

        # Handle parameter types
        if param_type == "Categorical":
            config_dict[name] = CategoricalRange(choices=choices)
        elif param_type == "UniformInteger":
            values = [int(x) for x in range_values]
            config_dict[name] = IntRange(type="int", lower=values[0], upper=values[1])
        elif param_type == "UniformFloat":
            values = [float(x) for x in range_values]
            config_dict[name] = FloatRange(
                type="float", lower=values[0], upper=values[1]
            )
        elif param_type == "Constant":
            config_dict[name] = CategoricalRange(choices=[value])

    config_dict["OpenML_task_id"] = CategoricalRange(choices=[openml_id])

    return config_dict
