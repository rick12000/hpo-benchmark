from confopt.selection.acquisition import (
    QuantileConformalSearcher,
)
from confopt.selection.sampling.bound_samplers import (
    LowerBoundSampler,
)
from confopt.selection.sampling.entropy_samplers import MaxValueEntropySearchSampler
from confopt.selection.sampling.expected_improvement_samplers import (
    ExpectedImprovementSampler,
)
from confopt.selection.sampling.thompson_samplers import ThompsonSampler
from hpobench.config.utils import (
    get_external_tuning_configurations,
    build_sampler_variation_configurations,
    build_architecture_variation_configurations,
)
from hpobench.config.config_types import (
    TunerConfig,
)

N_TRIALS = 100
TIMEOUT = None
N_WARM_STARTS = 15

# Environment variables used only in configuration:
DEFAULT_INTERVAL_WIDTH = 0.9

# 1. Create configurations feeding the static tuning charts and tables:
STATIC_ANALYSIS_ESTIMATOR_ARCHITECTURES = [
    "qgp",
    "ql",
    "qrf",
    "qgbm",
    "qens3",
    # "qens4",
]

# 2. Create configurations feeding the coverage charts:
COVERAGE_ANALYSIS_CONFIGURATIONS = []
COVERAGE_INTERVAL_WIDTHS = [0.25, 0.5, 0.75]  # 0.25, 0.5, 0.75
ADAPTERS = ["ACI", "DtACI", None]

for interval_width in COVERAGE_INTERVAL_WIDTHS:
    for adapter in ADAPTERS:
        SAMPLER = LowerBoundSampler(
            interval_width=interval_width,
            adapter=adapter,
            c=0,
        )
        SEARCHER = QuantileConformalSearcher(
            quantile_estimator_architecture="qgbm",
            sampler=SAMPLER,
            n_calibration_folds=5,
            calibration_split_strategy="train_test_split",
            symmetric_adjustment=True,
        )
        if adapter is None:
            config_identifier = "Conformalized"
        elif adapter in ["ACI", "DtACI"]:
            config_identifier = f"Conformalized + {adapter}"
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

        # SEARCHER = QuantileConformalSearcher(
        #     quantile_estimator_architecture="qgbm",
        #     sampler=SAMPLER,
        #     n_calibration_folds=5,
        #     calibration_split_strategy="cv",
        #     symmetric_adjustment=True,
        # )
        # if adapter is None:
        #     config_identifier = "Cross Validated"
        # elif adapter in ["ACI", "DtACI"]:
        #     config_identifier = f"Cross Validated + {adapter}"
        # else:
        #     raise ValueError(f"Unknown adapter: {adapter}")
        # COVERAGE_ANALYSIS_CONFIGURATIONS.append(
        #     TunerConfig(
        #         tuner="confopt",
        #         searcher=SEARCHER,
        #         config_identifier=config_identifier,
        #         searcher_tuning_framework=None,
        #     )
        # )

    # Manually add the unconformalized configuration for each interval width:
    COVERAGE_ANALYSIS_CONFIGURATIONS.append(
        TunerConfig(
            tuner="confopt",
            searcher=QuantileConformalSearcher(
                quantile_estimator_architecture="qgbm",
                sampler=LowerBoundSampler(
                    interval_width=interval_width,
                    adapter=None,
                    c=0,
                ),
                n_pre_conformal_trials=10000,
                n_calibration_folds=3,
                calibration_split_strategy="train_test_split",
                symmetric_adjustment=True,
            ),
            config_identifier="Unconformalized",
            searcher_tuning_framework=None,
        )
    )

# 3. Create configurations feeding the comparative tuner rank plots:
SAMPLER_VARIATION_N_DEFAULT_QUANTILES = 4
SAMPLER_VARIATION_DEFAULT_ADAPTER = "DtACI"
SAMPLER_VARIATION_CONFIGURATIONS = build_sampler_variation_configurations(
    samplers=[
        MaxValueEntropySearchSampler(
            n_quantiles=SAMPLER_VARIATION_N_DEFAULT_QUANTILES,
            adapter=SAMPLER_VARIATION_DEFAULT_ADAPTER,
            n_paths=1000,
            n_y_candidates_per_x=100,  # Should be 1000, but too slow
            entropy_method="distance",
        ),
        # LowerBoundSampler(
        #     interval_width=DEFAULT_INTERVAL_WIDTH,
        #     adapter=SAMPLER_VARIATION_DEFAULT_ADAPTER,
        #     c=2,
        #     beta_decay="logarithmic_decay",
        # ),
        # LowerBoundSampler(
        #     interval_width=DEFAULT_INTERVAL_WIDTH,
        #     adapter=SAMPLER_VARIATION_DEFAULT_ADAPTER,
        #     c=1,
        #     beta_decay="logarithmic_decay",
        # ),
        # PessimisticLowerBoundSampler(
        #     interval_width=DEFAULT_INTERVAL_WIDTH,
        #     adapter=SAMPLER_VARIATION_DEFAULT_ADAPTER,
        # ),
        # ExpectedImprovementSampler(
        #     n_quantiles=SAMPLER_VARIATION_N_DEFAULT_QUANTILES,
        #     num_ei_samples=1000,
        #     adapter=SAMPLER_VARIATION_DEFAULT_ADAPTER,
        # ),
        ThompsonSampler(
            n_quantiles=SAMPLER_VARIATION_N_DEFAULT_QUANTILES,
            enable_optimistic_sampling=False,
            adapter=SAMPLER_VARIATION_DEFAULT_ADAPTER,
        ),
        ThompsonSampler(
            n_quantiles=SAMPLER_VARIATION_N_DEFAULT_QUANTILES,
            enable_optimistic_sampling=True,
            adapter=SAMPLER_VARIATION_DEFAULT_ADAPTER,
        ),
    ],
    quantile_arch="qgbm",
)

ARCHITECTURE_VARIATION_ADAPTER = "DtACI"
ARCHITECTURE_VARIATION_N_QUANTILES = 4
ARCHITECTURE_VARIATION_CONFIGURATIONS = build_architecture_variation_configurations(
    architectures=[
        # "qgp",
        "ql",
        # "qrf",
        "qgbm",
        "qens3",
        # "qens4",
    ],
    samplers=[
        # ExpectedImprovementSampler(
        #     n_quantiles=ARCHITECTURE_VARIATION_N_QUANTILES,
        #     num_ei_samples=1000,
        #     adapter=ARCHITECTURE_VARIATION_ADAPTER,
        # ),
        ThompsonSampler(
            n_quantiles=ARCHITECTURE_VARIATION_N_QUANTILES,
            enable_optimistic_sampling=True,
            adapter=ARCHITECTURE_VARIATION_ADAPTER,
        ),
        # MaxValueEntropySearchSampler(
        #     n_quantiles=ARCHITECTURE_VARIATION_N_QUANTILES,
        #     adapter=ARCHITECTURE_VARIATION_ADAPTER,
        #     n_paths=1000,
        #     n_y_candidates_per_x=100,  # Should be 1000, but too slow
        #     entropy_method="distance",
        # ),
    ],
)

LIMITED_ARCHITECTURE_ADAPTER = "DtACI"
LIMITED_ARCHITECTURE_N_QUANTILES = 4
LIMITED_ARCHITECTURE_VARIATION_CONFIGURATIONS = build_architecture_variation_configurations(
    architectures=[
        # "qrf",
        "qgp",
        # "ql",
        # "qgbm",
        # "qens1",
        # "qens2",
        # "qens3",
        # "qens4",
        # "qens5",
    ],
    samplers=[
        ThompsonSampler(
            n_quantiles=LIMITED_ARCHITECTURE_N_QUANTILES,
            enable_optimistic_sampling=True,
            adapter=LIMITED_ARCHITECTURE_ADAPTER,
        )
    ],
    n_pre_conformal_trials=32,
    searcher_tuning_framework=None,
)


PRECONFORMAL_ADAPTER = "DtACI"
PRECONFORMAL_N_QUANTILES = 4
PRECONFORMAL_COMPARISON_CONFIGURATIONS = []
for architecture in [
    # "qgp",
    "ql",
    # "qgbm",
    "qrf",
]:
    # Simulate normal pre-conformal cutoff vs. unreachable one:
    for pre_conformal_trials in [32, 10000]:
        if pre_conformal_trials == 10000:
            adapter = None
        else:
            adapter = PRECONFORMAL_ADAPTER
        PRECONFORMAL_COMPARISON_CONFIGURATIONS.extend(
            build_architecture_variation_configurations(
                architectures=[architecture],
                samplers=[
                    ExpectedImprovementSampler(
                        n_quantiles=PRECONFORMAL_N_QUANTILES,
                        num_ei_samples=1000,
                        adapter=adapter,
                    ),
                    # MaxValueEntropySearchSampler(
                    #     n_quantiles=PRECONFORMAL_N_QUANTILES,
                    #     adapter=adapter,
                    #     n_paths=1000,
                    #     n_y_candidates_per_x=100,
                    #     entropy_method="distance",
                    # ),
                    ThompsonSampler(
                        n_quantiles=PRECONFORMAL_N_QUANTILES,
                        enable_optimistic_sampling=True,
                        adapter=adapter,
                    ),
                ],
                n_pre_conformal_trials=pre_conformal_trials,
            )
        )


# 4. Create configurations feeding the quantile count variation plots:
QUANTILE_COUNT_VARIATION_ADAPTER = "DtACI"
QUANTILE_COUNT_VARIATION_CONFIGURATIONS = []
QUANTILE_COUNT_VALUES = [4, 10]

for n_quantiles in QUANTILE_COUNT_VALUES:
    QUANTILE_COUNT_VARIATION_CONFIGURATIONS.extend(
        build_architecture_variation_configurations(
            architectures=[
                "qrf",  # Use single architecture
            ],
            samplers=[
                ThompsonSampler(
                    n_quantiles=n_quantiles,
                    enable_optimistic_sampling=True,
                    adapter=QUANTILE_COUNT_VARIATION_ADAPTER,
                ),
                ExpectedImprovementSampler(
                    n_quantiles=n_quantiles,
                    num_ei_samples=1000,
                    adapter=QUANTILE_COUNT_VARIATION_ADAPTER,
                ),
                # MaxValueEntropySearchSampler(
                #     n_quantiles=n_quantiles,
                #     adapter=QUANTILE_COUNT_VARIATION_ADAPTER,
                #     n_paths=1000,
                #     n_y_candidates_per_x=100,
                #     entropy_method="distance",
                # ),
            ],
            n_pre_conformal_trials=32,
            searcher_tuning_framework=None,
        )
    )


# 5. Create configurations feeding the search tuning effect plots:
SEARCH_TUNING_EFFECT_ADAPTER = "DtACI"
SEARCH_TUNING_EFFECT_N_QUANTILES = 4
SEARCH_TUNING_EFFECT_CONFIGURATIONS = []

# Use multiple architectures and vary searcher_tuning_framework (None vs "fixed")
for searcher_tuning_framework in [None, "fixed"]:
    SEARCH_TUNING_EFFECT_CONFIGURATIONS.extend(
        build_architecture_variation_configurations(
            architectures=[
                "ql",
                "qrf",
                "qgbm",
                "qens3",
            ],
            samplers=[
                ThompsonSampler(
                    n_quantiles=SEARCH_TUNING_EFFECT_N_QUANTILES,
                    enable_optimistic_sampling=True,
                    adapter=SEARCH_TUNING_EFFECT_ADAPTER,
                )
            ],
            n_pre_conformal_trials=32,
            searcher_tuning_framework=searcher_tuning_framework,
        )
    )


EXTERNAL_TUNING_CONFIGURATIONS = get_external_tuning_configurations()
