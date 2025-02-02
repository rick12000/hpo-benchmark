import random

synthetic_params = {
    "param1__range_float": [0, 100],
    "param2__range_float": [0, 100],
    "param3__range_float": [0, 100],
    "param4__range_float": [0, 100],
    "param5__range_float": [0, 100],
    "param6__range_float": [0, 100],
    "param7__range_float": [0, 100],
}


def generate_hyperparameter_combinations(confopt_params, n_combinations, random_state):
    random.seed(random_state)
    combinations = []
    for _ in range(n_combinations):
        combination = {}
        for param_name, param_values in confopt_params.items():
            if "__range_int" in param_name:
                combination[param_name.replace("__range_int", "")] = list(
                    range(param_values[0], param_values[1] + 1)
                )
            elif "__range_float" in param_name:
                combination[param_name.replace("__range_float", "")] = [
                    random.uniform(param_values[0], param_values[1])
                    for _ in range(1000)
                ]
            else:
                combination[param_name] = param_values
        combinations.append(combination)
    return combinations

    random.seed(random_state)


generate_hyperparameter_combinations(
    confopt_params=synthetic_params, n_combinations=5, random_state=1
)
