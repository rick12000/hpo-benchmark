from typing import Literal
from confopt.selection.acquisition import (
    QuantileConformalSearcher,
)
from confopt.selection.sampling import (
    ThompsonSampler,
    LowerBoundSampler,
    ExpectedImprovementSampler,
    InformationGainSampler,
    MaxValueEntropySearchSampler,
)
from hpobench.config.utils import (
    create_sampler_config_id,
    build_static_tuning_configurations,
    get_external_tuning_configurations,
    build_sampler_variation_configurations,
    build_architecture_variation_configurations,
)
from hpobench.config.types import (
    TunerConfig,
)

# Environment variables used in the main code:
N_REPETITIONS_PER_TUNER_CONFIG = 20
N_TRIALS = 60
TIMEOUT = None
N_WARM_STARTS = 15
RUN_TYPE: Literal["dev", "full"] = "full"

# Environment variables used only in configuration:
DEFAULT_INTERVAL_WIDTH = 0.9

# 1. Create configurations feeding the static tuning charts and tables:
STATIC_TUNING_CONFIGURATIONS = build_static_tuning_configurations(
    quantile_architectures=[
        "qgbm",
        "qrf",
        "qknn",
        "qlgbm",
        "qgp",
        "qens4",
    ],
    searcher_tuning_frameworks=[None, "fixed"],
)

# 2. Create configurations feeding the coverage charts:
COVERAGE_ANALYSIS_CONFIGURATIONS = []
# TODO: Add a fixed DtACI to simulate ACI:
ADAPTERS = ["DtACI", None]
for adapter in ADAPTERS:
    SAMPLER = LowerBoundSampler(
        interval_width=DEFAULT_INTERVAL_WIDTH,
        adapter=adapter,
        c=1,
    )
    SEARCHER = QuantileConformalSearcher(
        quantile_estimator_architecture="qrf",
        sampler=SAMPLER,
    )
    if adapter is None:
        config_identifier = "Conformalized"
    elif adapter == "DtACI":
        config_identifier = "Conformalized + DtACI"
    else:
        raise ValueError(f"Unknown adapter: {adapter}")
    COVERAGE_ANALYSIS_CONFIGURATIONS.append(
        TunerConfig(
            tuner="confopt",
            searcher=SEARCHER,
            config_identifier=config_identifier,
            searcher_tuning_framework=None,
        )
    )
# Manually add the unconformalized configuration:
COVERAGE_ANALYSIS_CONFIGURATIONS.append(
    TunerConfig(
        tuner="confopt",
        searcher=QuantileConformalSearcher(
            quantile_estimator_architecture="qrf",
            sampler=SAMPLER,
            n_pre_conformal_trials=10000,
        ),
        config_identifier="Unconformalized",
        searcher_tuning_framework=None,
    )
)

# 3. Create configurations feeding the comparative tuner rank plots:
SAMPLER_VARIATION_CONFIGURATIONS = build_sampler_variation_configurations(
    samplers=[
        InformationGainSampler(
            n_quantiles=8,
            adapter="DtACI",
            n_paths=100,
            n_X_candidates=10,
            n_y_candidates_per_x=4,
            sampling_strategy="thompson",
        ),
        MaxValueEntropySearchSampler(
            n_quantiles=8,
            adapter="DtACI",
            n_min_samples=100,
            n_y_samples=30,
            entropy_method="distance",
        ),
        LowerBoundSampler(
            interval_width=DEFAULT_INTERVAL_WIDTH,
            adapter="DtACI",
            c=2,
            beta_decay="logarithmic_decay",
        ),
        ExpectedImprovementSampler(n_quantiles=8, num_ei_samples=100, adapter="DtACI"),
        # ThompsonSampler(n_quantiles=8, enable_optimistic_sampling=False, adapter="DtACI"),
        ThompsonSampler(
            n_quantiles=8, enable_optimistic_sampling=True, adapter="DtACI"
        ),
    ],
    quantile_arch="qgbm",
)

ARCHITECTURE_VARIATION_CONFIGURATIONS = build_architecture_variation_configurations(
    architectures=[
        # "qrf",
        # "qknn",
        # "qens1",
        # "qgp",
        # "qens2",
        # "qens3",
        "qens4",
        # "qens5"
    ],
    samplers=[
        ExpectedImprovementSampler(n_quantiles=8, num_ei_samples=100, adapter=None),
        # ThompsonSampler(n_quantiles=8, enable_optimistic_sampling=False, adapter="DtACI"),
    ],
)

PRECONFORMAL_COMPARISON_CONFIGURATIONS = build_architecture_variation_configurations(
    architectures=["qgbm", "qgp"],
    samplers=[
        ExpectedImprovementSampler(n_quantiles=8, num_ei_samples=100, adapter=None)
    ],
    n_pre_conformal_trials=10000,
)
EXTERNAL_TUNING_CONFIGURATIONS = get_external_tuning_configurations()

FULL_TUNING_CONFIGURATIONS = [
    EXTERNAL_TUNING_CONFIGURATIONS,
    SAMPLER_VARIATION_CONFIGURATIONS,
    ARCHITECTURE_VARIATION_CONFIGURATIONS,
    PRECONFORMAL_COMPARISON_CONFIGURATIONS,
]

# 4. Create configurations feeding the tuning vs. no tuning search rank plots (NOTE: Unused in paper for brevity):
SAMPLER = LowerBoundSampler(
    interval_width=DEFAULT_INTERVAL_WIDTH,
    adapter="DtACI",
    c=1,
)
SEARCHER = QuantileConformalSearcher(
    quantile_estimator_architecture="qrf",
    sampler=SAMPLER,
)
TUNING_PATH_CONFIGURATIONS = [
    TunerConfig(
        tuner="confopt",
        searcher=SEARCHER,
        config_identifier=create_sampler_config_id(SEARCHER),
        searcher_tuning_framework=None,  # No tuning
    ),
    # Same configuration but with tuning enabled
    TunerConfig(
        tuner="confopt",
        searcher=SEARCHER,  # Reuse the same searcher
        config_identifier=create_sampler_config_id(
            SEARCHER, searcher_tuning_framework="fixed"
        ),
        searcher_tuning_framework="fixed",  # Fixed tuning framework
    ),
]

# 5. Create configurations for development use:
DEV_TUNING_CONFIGURATIONS = [
    TunerConfig(
        tuner="optuna",
        searcher="tpe",
        config_identifier="TPE",
    ),
    TunerConfig(
        tuner="confopt",
        searcher=QuantileConformalSearcher(
            quantile_estimator_architecture="qrf",
            sampler=MaxValueEntropySearchSampler(
                n_quantiles=8,
                adapter="DtACI",
                n_min_samples=500,
                n_y_samples=30,
            ),
        ),
        config_identifier=create_sampler_config_id(
            QuantileConformalSearcher(
                quantile_estimator_architecture="qrf",
                sampler=MaxValueEntropySearchSampler(
                    n_quantiles=8,
                    adapter="DtACI",
                    n_min_samples=500,
                    n_y_samples=30,
                ),
            )
        ),
        searcher_tuning_framework=None,
    ),
    TunerConfig(
        tuner="confopt",
        searcher=QuantileConformalSearcher(
            quantile_estimator_architecture="qrf",
            sampler=ThompsonSampler(
                n_quantiles=8, enable_optimistic_sampling=False, adapter="DtACI"
            ),
        ),
        config_identifier=create_sampler_config_id(
            QuantileConformalSearcher(
                quantile_estimator_architecture="qrf",
                sampler=ThompsonSampler(
                    n_quantiles=8, enable_optimistic_sampling=False, adapter="DtACI"
                ),
            )
        ),
        searcher_tuning_framework=None,
    ),
]
