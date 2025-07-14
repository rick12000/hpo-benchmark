from hpobench.config.types import (
    FloatRange,
    CategoricalRange,
)


JAHS201_SEARCH_SPACE = {
    "Activation": CategoricalRange(choices=["ReLU", "Hardswish", "Mish"]),
    "LearningRate": FloatRange(lower=0.001, upper=1),
    "WeightDecay": FloatRange(lower=1e-5, upper=1e-2),
    "Op1": CategoricalRange(choices=list(range(5))),
    "Op2": CategoricalRange(choices=list(range(5))),
    "Op3": CategoricalRange(choices=list(range(5))),
    "Op4": CategoricalRange(choices=list(range(5))),
    "Op5": CategoricalRange(choices=list(range(5))),
    "Op6": CategoricalRange(choices=list(range(5))),
    "Optimizer": CategoricalRange(choices=["SGD"]),
    "TrivialAugment": CategoricalRange(choices=[True, False]),
}


n_synthetic_params = 10
BLACK_BOX_SEARCH_SPACE = {}
for n in range(n_synthetic_params):
    BLACK_BOX_SEARCH_SPACE[f"param{n}"] = FloatRange(lower=0, upper=100)
BLACK_BOX_IDS: list[str] = ["rastrigin", "shekel", "weierstrass", "griewank", "ackley"]
