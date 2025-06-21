from confopt.selection.acquisition import (
    QuantileConformalSearcher,
)
from confopt.selection.sampling import (
    ThompsonSampler,
    LowerBoundSampler,
    ExpectedImprovementSampler,
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
N_REPETITIONS_PER_TUNER_CONFIG = 3
N_TRIALS = 100
TIMEOUT = None
N_WARM_STARTS = 15

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
COVERAGE_INTERVAL_WIDTHS = [0.1, 0.5, 0.9]
ADAPTERS = ["ACI", "DtACI", None]

for interval_width in COVERAGE_INTERVAL_WIDTHS:
    for adapter in ADAPTERS:
        SAMPLER = LowerBoundSampler(
            interval_width=interval_width,
            adapter=adapter,
            c=1,
        )
        SEARCHER = QuantileConformalSearcher(
            quantile_estimator_architecture="qrf",
            sampler=SAMPLER,
        )
        if adapter is None:
            config_identifier = f"Conformalized @ {interval_width}%"
        elif adapter in ["ACI", "DtACI"]:
            config_identifier = f"Conformalized + {adapter} @ {interval_width}%"
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
    # Manually add the unconformalized configuration for each interval width:
    COVERAGE_ANALYSIS_CONFIGURATIONS.append(
        TunerConfig(
            tuner="confopt",
            searcher=QuantileConformalSearcher(
                quantile_estimator_architecture="qrf",
                sampler=LowerBoundSampler(
                    interval_width=interval_width,
                    adapter=None,
                    c=1,
                ),
                n_pre_conformal_trials=10000,
            ),
            config_identifier=f"Unconformalized @ {interval_width}%",
            searcher_tuning_framework=None,
        )
    )

# 3. Create configurations feeding the comparative tuner rank plots:
SAMPLER_VARIATION_CONFIGURATIONS = build_sampler_variation_configurations(
    samplers=[
        #     InformationGainSampler(
        #         n_quantiles=8,
        #         adapter="DtACI",
        #         n_paths=100,
        #         n_X_candidates=10,
        #         n_y_candidates_per_x=4,
        #         sampling_strategy="thompson",
        #     ),
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
        "qrf",
        "qknn",
        # "qens1",
        # "qgp",
        # "qens2",
        # "qens3",
        "qgbm",
        "qens4",
        # "qens5"
    ],
    samplers=[
        ExpectedImprovementSampler(n_quantiles=8, num_ei_samples=100, adapter="DtACI"),
        ThompsonSampler(
            n_quantiles=8, enable_optimistic_sampling=False, adapter="DtACI"
        ),
    ],
)


LIMITED_ARCHITECTURE_VARIATION_CONFIGURATIONS = (
    build_architecture_variation_configurations(
        architectures=[
            "qrf",
            "qgp",
            "qens4",
        ],
        samplers=[
            ExpectedImprovementSampler(
                n_quantiles=8, num_ei_samples=100, adapter="DtACI"
            )
        ],
    )
)

PRECONFORMAL_COMPARISON_CONFIGURATIONS = []
for architecture in ["qgbm", "qgp", "qens4"]:
    # Simulate normal pre-conformal cutoff vs. unreachable one:
    for pre_conformal_trials in [20, 10000]:
        if pre_conformal_trials == 10000:
            adapter = None
        else:
            adapter = "DtACI"
        PRECONFORMAL_COMPARISON_CONFIGURATIONS.extend(
            build_architecture_variation_configurations(
                architectures=[architecture],
                samplers=[
                    ExpectedImprovementSampler(
                        n_quantiles=8, num_ei_samples=100, adapter=adapter
                    )
                ],
                n_pre_conformal_trials=pre_conformal_trials,
            )
        )


EXTERNAL_TUNING_CONFIGURATIONS = get_external_tuning_configurations()

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
