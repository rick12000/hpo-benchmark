from pydantic import BaseModel, ConfigDict, root_validator
from typing import Union, Literal, Optional
from confopt.selection.acquisition import (
    QuantileConformalSearcher,
    LocallyWeightedConformalSearcher,
)
from copy import deepcopy
from confopt.selection.sampling import (
    ThompsonSampler,
    LowerBoundSampler,
    ExpectedImprovementSampler,
    InformationGainSampler,
    MaxValueEntropySearchSampler,
)

from hpobench.generate import ObjectiveMetricGenerator


def create_sampler_config_id(
    searcher, custom_prefix=None, searcher_tuning_framework=None
):
    """Create a configuration ID string based on searcher properties.

    Args:
        searcher: A QuantileConformalSearcher instance
        custom_prefix: Optional prefix to add to the config ID
        searcher_tuning_framework: Optional indicator for tuning framework

    Returns:
        A formatted string for the configuration ID
    """
    # Start with custom prefix if provided
    config_id = f"{custom_prefix}-" if custom_prefix else ""

    # Handle string-based searchers (like "tpe")
    if isinstance(searcher, str):
        return searcher.upper()

    # Extract relevant information
    if not hasattr(searcher, "quantile_estimator_architecture"):
        raise ValueError(
            "Input must be a QuantileConformalSearcher instance or a string"
        )

    sampler = searcher.sampler
    quantile_arch = searcher.quantile_estimator_architecture
    n_pre_conformal_trials = (
        searcher.n_pre_conformal_trials
        if hasattr(searcher, "n_pre_conformal_trials")
        else None
    )

    # Extract sampler acronym from class name (uppercase letters only)
    sampler_class_name = sampler.__class__.__name__
    sampler_acronym = "".join(c for c in sampler_class_name if c.isupper())

    # Extract adapter name
    adapter_name = None
    if hasattr(sampler, "adapters") and sampler.adapters:
        adapter_name = sampler.adapters[0].__class__.__name__
    elif hasattr(sampler, "adapter"):
        if isinstance(sampler.adapter, str):
            adapter_name = sampler.adapter
        elif sampler.adapter is not None:
            adapter_name = sampler.adapter.__class__.__name__

    # Build the config ID
    if quantile_arch:
        quantile_arch_upper = quantile_arch.upper()
        if adapter_name:
            config_id += f"{adapter_name}-{quantile_arch_upper} {sampler_acronym}"
        else:
            config_id += f"{quantile_arch_upper} {sampler_acronym}"
    else:
        # For non-quantile-based methods
        config_id += f"{sampler_acronym}"

    # Add additional attributes based on sampler type
    if hasattr(sampler, "c"):
        config_id += f" c={sampler.c}"

    if hasattr(sampler, "beta_decay") and sampler.beta_decay:
        # Convert snake_case to acronym (first letter of each word)
        decay_parts = sampler.beta_decay.split("_")
        decay_acronym = "".join(part[0] for part in decay_parts)
        config_id += f" {decay_acronym}"

    if (
        hasattr(sampler, "enable_optimistic_sampling")
        and sampler.enable_optimistic_sampling
    ):
        config_id += " OPT"

    if hasattr(sampler, "n_quantiles"):
        config_id += f" nq={sampler.n_quantiles}"

    if hasattr(sampler, "num_ei_samples"):
        config_id += f" ns={sampler.num_ei_samples}"

    # Add pre-conformal trials if specified and not default
    if n_pre_conformal_trials and n_pre_conformal_trials != 20:
        config_id += f" pre={n_pre_conformal_trials}"

    # Add tuning framework suffix based on type
    if searcher_tuning_framework == "fixed":
        config_id += " TUNED-F"
    elif searcher_tuning_framework == "reward_cost":
        config_id += " TUNED-A"
    elif searcher_tuning_framework:  # Any other non-None value
        config_id += " TUNED"

    return config_id


N_REPETITIONS_PER_TUNER_CONFIG = 10
N_TRIALS = 85
TIMEOUT = None
N_WARM_STARTS = 15
RUN_TYPE: Literal["dev", "full"] = "full"


class TunerConfig(BaseModel):
    tuner: Literal["confopt", "optuna", "skopt"]
    searcher: Union[
        str,
        Literal["tpe", "random", "cmaes", "gbrt", "forest", "gp"],
        QuantileConformalSearcher,
        LocallyWeightedConformalSearcher,
    ]
    config_identifier: str
    searcher_tuning_framework: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class FloatRange(BaseModel):
    type: str = "float"
    lower: float
    upper: float


class IntRange(BaseModel):
    type: str = "int"
    lower: int
    upper: int


class CategoricalRange(BaseModel):
    type: str = "categorical"
    choices: list[Union[str, int, bool]]


class ExperimentConfig(BaseModel):
    search_space: dict[str, Union[IntRange, FloatRange, CategoricalRange]]
    generator: ObjectiveMetricGenerator
    tuning_configurations: list[TunerConfig]
    n_warm_starts: int
    benchmark_identifier: str
    dataset_identifier: str
    metric: Optional[str] = None
    n_trials: Optional[int] = None
    timeout: Optional[float] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @root_validator(pre=True)
    def check_timeout_or_n_trials(cls, values):
        if values.get("n_trials") is None and values.get("timeout") is None:
            raise ValueError(
                "At least one of 'n_trials' or 'timeout' must be specified."
            )
        return values


default_interval_width = 0.9
default_sampler = LowerBoundSampler(
    interval_width=default_interval_width,
    adapter="DtACI",
    c=1,
    beta_decay="logarithmic_decay",
)

_QUANTILE_ARCHITECTURES = [
    # "qgbm",
    # "qrf",
    # "qknn",
    # "ql",
    # "qlgbm",
    "qgp",
    "qens1",
    # "qens2",
    # "qens3",
    "qens4",
    "qens5",
]
_FRAMEWORKS = [
    (None, ""),
    ("fixed", " TUNED-F"),
]
STATIC_TUNING_CONFIGURATIONS: list[TunerConfig] = [
    TunerConfig(
        tuner="confopt",
        searcher=QuantileConformalSearcher(
            quantile_estimator_architecture=arch,
            sampler=deepcopy(default_sampler),
        ),
        config_identifier=create_sampler_config_id(
            QuantileConformalSearcher(
                quantile_estimator_architecture=arch,
                sampler=deepcopy(default_sampler),
            ),
            searcher_tuning_framework=framework,
        ),
        searcher_tuning_framework=framework,
    )
    for arch in _QUANTILE_ARCHITECTURES
    for framework, suffix in _FRAMEWORKS
]


JAHS201_SEARCH_SPACE = {
    "Activation": CategoricalRange(
        type="categorical", choices=["ReLU", "Hardswish", "Mish"]
    ),
    "LearningRate": FloatRange(type="float", lower=0.001, upper=1),
    "N": CategoricalRange(type="categorical", choices=[5]),
    "Op1": CategoricalRange(type="categorical", choices=list(range(5))),
    "Op2": CategoricalRange(type="categorical", choices=list(range(5))),
    "Op3": CategoricalRange(type="categorical", choices=list(range(5))),
    "Op4": CategoricalRange(type="categorical", choices=list(range(5))),
    "Op5": CategoricalRange(type="categorical", choices=list(range(5))),
    "Op6": CategoricalRange(type="categorical", choices=list(range(5))),
    "Optimizer": CategoricalRange(type="categorical", choices=["SGD"]),
    "Resolution": CategoricalRange(type="categorical", choices=[1]),
    "TrivialAugment": CategoricalRange(type="categorical", choices=[True, False]),
    "W": CategoricalRange(type="categorical", choices=[16]),
    "WeightDecay": FloatRange(type="float", lower=0.00001, upper=0.01),
    "epoch": CategoricalRange(type="categorical", choices=[200]),
}


n_synthetic_params = 10
BLACK_BOX_SEARCH_SPACE = {}
for n in range(n_synthetic_params):
    BLACK_BOX_SEARCH_SPACE[f"param{n}"] = FloatRange(type="float", lower=0, upper=100)

JAHS201_IDS: list[str] = ["cifar10", "fashion_mnist", "colorectal_histology"]
BLACK_BOX_IDS: list[str] = ["rastrigin", "shekel", "weierstrass", "griewank", "ackley"]


FULL_TUNING_CONFIGURATIONS = [
    # TunerConfig(
    #     tuner="confopt",
    #     searcher=QuantileConformalSearcher(
    #         quantile_estimator_architecture="qrf",
    #         sampler= MaxValueEntropySearchSampler(n_quantiles=8, adapter="DtACI",n_min_samples=500, n_y_samples=30, alpha=0.05, sampling_strategy="uniform"),
    #     ),
    #     config_identifier=create_sampler_config_id(
    #         QuantileConformalSearcher(
    #             quantile_estimator_architecture="qrf",
    #             sampler=MaxValueEntropySearchSampler(n_quantiles=8, adapter="DtACI", n_min_samples=500, n_y_samples=30, alpha=0.05, sampling_strategy="uniform"),
    #         )
    #     ),
    #     searcher_tuning_framework=None,
    # ),
    # # TunerConfig with commented out sampler removed for brevity
    # TunerConfig(
    #     tuner="confopt",
    #     searcher=QuantileConformalSearcher(
    #         quantile_estimator_architecture="qrf",
    #         sampler= ThompsonSampler(n_quantiles=8, enable_optimistic_sampling=False, adapter="DtACI"),
    #     ),
    #     config_identifier=create_sampler_config_id(
    #         QuantileConformalSearcher(
    #             quantile_estimator_architecture="qrf",
    #             sampler=ThompsonSampler(n_quantiles=8, enable_optimistic_sampling=False, adapter="DtACI"),
    #         )
    #     ),
    #     searcher_tuning_framework=None,
    # ),
]

# Define a list of samplers with their parameters
SAMPLERS = [
    # InformationGainSampler(
    #     n_quantiles=8,
    #     adapter="DtACI",
    #     n_paths=100,
    #     n_X_candidates=10,
    #     n_y_candidates_per_x=4,
    #     sampling_strategy="thompson"
    # ),
    # MaxValueEntropySearchSampler(n_quantiles=8, adapter="DtACI", n_min_samples=500, n_y_samples=30, sampling_strategy="uniform", entropy_method = "distance"),
    # LowerBoundSampler with logarithmic decay
    LowerBoundSampler(
        interval_width=default_interval_width,
        adapter="DtACI",
        c=2,
        beta_decay="logarithmic_decay",
    ),
    # ExpectedImprovementSampler
    ExpectedImprovementSampler(n_quantiles=8, num_ei_samples=100, adapter="DtACI"),
    # ThompsonSampler (regular)
    ThompsonSampler(n_quantiles=8, enable_optimistic_sampling=False, adapter="DtACI"),
    # ThompsonSampler (optimistic)
    ThompsonSampler(n_quantiles=8, enable_optimistic_sampling=True, adapter="DtACI"),
]

# Define quantile estimator architecture to use for all samplers
QUANTILE_ARCH = "qrf"

# Create configurations systematically
SAMPLER_VARIATION_CONFIGURATIONS = []
for sampler in SAMPLERS:
    # Create searcher instance
    searcher = QuantileConformalSearcher(
        quantile_estimator_architecture=QUANTILE_ARCH,
        sampler=sampler,
    )

    # Create the config ID using the new function
    config_id = create_sampler_config_id(searcher)

    # Create and append the TunerConfig
    SAMPLER_VARIATION_CONFIGURATIONS.append(
        TunerConfig(
            tuner="confopt",
            searcher=searcher,
            config_identifier=config_id,
            searcher_tuning_framework=None,
        )
    )

# Append the sampler variations to the full configurations
# FULL_TUNING_CONFIGURATIONS.extend(SAMPLER_VARIATION_CONFIGURATIONS)

# Create configurations that vary the quantile estimator architecture while keeping the sampler fixed
ARCHITECTURE_VARIATION_CONFIGURATIONS = []

# Define architectures to loop through
ARCHITECTURE_LIST = [
    "qrf",
    # "qknn",
    "qens1",
    "qgp",
    #  "qens2",
    #  "qens3",
    #  "qens4",
    #  "qens5"
]

# Create a fixed LowerBoundSampler with explicit parameters
fixed_sampler = LowerBoundSampler(
    interval_width=default_interval_width,
    adapter="DtACI",
    c=1,
    beta_decay="logarithmic_decay",
    beta_max=10.0,
)
# fixed_sampler = ThompsonSampler(n_quantiles=8, enable_optimistic_sampling=False, adapter="DtACI")

# Loop through architectures and create configurations
for arch in ARCHITECTURE_LIST:
    # Create searcher instance
    searcher = QuantileConformalSearcher(
        quantile_estimator_architecture=arch,
        sampler=deepcopy(fixed_sampler),
        n_pre_conformal_trials=20,
    )

    # Create a descriptive config ID
    config_id = create_sampler_config_id(searcher, custom_prefix="ArchVar")

    # Create and append the TunerConfig
    ARCHITECTURE_VARIATION_CONFIGURATIONS.append(
        TunerConfig(
            tuner="confopt",
            searcher=searcher,
            config_identifier=config_id,
            searcher_tuning_framework=None,
        )
    )

# Append the architecture variations to the full configurations
FULL_TUNING_CONFIGURATIONS.extend(ARCHITECTURE_VARIATION_CONFIGURATIONS)


# Create configurations that compare preconformal trials and adapters
PRECONFORMAL_COMPARISON_CONFIGURATIONS = []

# List of quantile estimator architectures to test
PRECONFORMAL_ARCH_LIST = ["qgbm", "qrf", "qknn", "ql", "qgp", "qens1", "qens4"]

# For each architecture, create a pair of configurations
for arch in PRECONFORMAL_ARCH_LIST:
    # Configuration with 10,000 preconformal trials and no adapter
    no_adapter_searcher = QuantileConformalSearcher(
        quantile_estimator_architecture=arch,
        sampler=LowerBoundSampler(
            interval_width=default_interval_width,
            adapter=None,
            c=1,
            beta_decay="logarithmic_decay",
            beta_max=10.0,
        ),
        n_pre_conformal_trials=10000,
    )

    PRECONFORMAL_COMPARISON_CONFIGURATIONS.append(
        TunerConfig(
            tuner="confopt",
            searcher=no_adapter_searcher,
            config_identifier=create_sampler_config_id(no_adapter_searcher),
            searcher_tuning_framework=None,
        )
    )

    # Configuration with default preconformal trials and DtACI adapter
    dt_aci_searcher = QuantileConformalSearcher(
        quantile_estimator_architecture=arch,
        sampler=LowerBoundSampler(
            interval_width=default_interval_width,
            adapter="DtACI",
            c=1,
            beta_decay="logarithmic_decay",
            beta_max=10.0,
        ),
        n_pre_conformal_trials=20,
    )

    PRECONFORMAL_COMPARISON_CONFIGURATIONS.append(
        TunerConfig(
            tuner="confopt",
            searcher=dt_aci_searcher,
            config_identifier=create_sampler_config_id(dt_aci_searcher),
            searcher_tuning_framework=None,
        )
    )

# Append the preconformal comparison configurations to the full configurations
# FULL_TUNING_CONFIGURATIONS.extend(PRECONFORMAL_COMPARISON_CONFIGURATIONS)


# Configurations specifically for dataset-level benchmarks
# One with tuning disabled and one with tuning enabled using same estimator architecture
dataset_benchmark_sampler = LowerBoundSampler(
    interval_width=default_interval_width,
    adapter="DtACI",
    c=1,
)

# Configuration with tuning disabled
qrf_searcher_no_tuning = QuantileConformalSearcher(
    quantile_estimator_architecture="qrf",
    sampler=dataset_benchmark_sampler,
)

DATASET_BENCHMARK_TUNING_CONFIGURATIONS = [
    TunerConfig(
        tuner="confopt",
        searcher=qrf_searcher_no_tuning,
        config_identifier=create_sampler_config_id(qrf_searcher_no_tuning),
        searcher_tuning_framework=None,  # No tuning
    ),
    # Same configuration but with tuning enabled
    TunerConfig(
        tuner="confopt",
        searcher=qrf_searcher_no_tuning,  # Reuse the same searcher
        config_identifier=create_sampler_config_id(
            qrf_searcher_no_tuning, searcher_tuning_framework="fixed"
        ),
        searcher_tuning_framework="fixed",  # Fixed tuning framework
    ),
]

DEV_TUNING_CONFIGURATIONS = [
    TunerConfig(
        tuner="optuna",
        searcher="tpe",
        config_identifier="TPE",  # Keep simple name for string-based searcher
    ),
    TunerConfig(
        tuner="confopt",
        searcher=QuantileConformalSearcher(
            quantile_estimator_architecture="qknn",
            sampler=LowerBoundSampler(interval_width=0.8, adapter="DtACI"),
        ),
        config_identifier=create_sampler_config_id(
            QuantileConformalSearcher(
                quantile_estimator_architecture="qknn",
                sampler=LowerBoundSampler(interval_width=0.8, adapter="DtACI"),
            )
        ),
        searcher_tuning_framework=None,
    ),
]

# Competing (non-confopt) configurations for comparison
COMPETING_TUNING_CONFIGURATIONS = [
    # TunerConfig(
    #     tuner="optuna",
    #     searcher="cmaes",
    #     config_identifier="CMA-ES",
    # ),
    TunerConfig(
        tuner="optuna",
        searcher="tpe",
        config_identifier="TPE",
    ),
    # TunerConfig(
    #     tuner="optuna",
    #     searcher="random",
    #     config_identifier="RS",
    # ),
    TunerConfig(
        tuner="optuna",
        searcher="gp",
        config_identifier="GP",
    ),
    # TunerConfig(
    #     tuner="skopt",
    #     searcher="gbrt",
    #     config_identifier="GBRT",
    # ),
]

# Append competing configurations to the full list
FULL_TUNING_CONFIGURATIONS.extend(COMPETING_TUNING_CONFIGURATIONS)

# Deduplicate configurations based on their config_identifier
seen_config_ids = set()
unique_tuning_configurations = []

for config in FULL_TUNING_CONFIGURATIONS:
    if config.config_identifier not in seen_config_ids:
        seen_config_ids.add(config.config_identifier)
        unique_tuning_configurations.append(config)
    # else this is a duplicate, skip it

# Replace the original list with the deduplicated one
FULL_TUNING_CONFIGURATIONS = unique_tuning_configurations
print(
    f"After deduplication: {len(FULL_TUNING_CONFIGURATIONS)} unique tuning configurations"
)
