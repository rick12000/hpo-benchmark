from confopt.selection.acquisition import (
    QuantileConformalSearcher,
)
from confopt.selection.sampling.bound_samplers import (
    LowerBoundSampler,
    PessimisticLowerBoundSampler,
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
from hpobench.config.types import (
    TunerConfig,
)

# Environment variables used in the main code:
N_REPETITIONS_PER_TUNER_CONFIG = 3
N_TRIALS = 50
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
    "qens4",
]

# 2. Create configurations feeding the coverage charts:
COVERAGE_ANALYSIS_CONFIGURATIONS = []
COVERAGE_INTERVAL_WIDTHS = [0.2, 0.4, 0.6, 0.8]  # , 0.9]
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
            ),
            config_identifier="Unconformalized",
            searcher_tuning_framework=None,
        )
    )

# 3. Create configurations feeding the comparative tuner rank plots:
SAMPLER_VARIATION_N_DEFAULT_QUANTILES = 10
SAMPLER_VARIATION_DEFAULT_ADAPTER = None
SAMPLER_VARIATION_CONFIGURATIONS = build_sampler_variation_configurations(
    samplers=[
        MaxValueEntropySearchSampler(
            n_quantiles=SAMPLER_VARIATION_N_DEFAULT_QUANTILES,
            adapter=SAMPLER_VARIATION_DEFAULT_ADAPTER,
            n_paths=1000,
            n_y_candidates_per_x=100,  # Should be 1000, but too slow
            entropy_method="distance",
        ),
        LowerBoundSampler(
            interval_width=DEFAULT_INTERVAL_WIDTH,
            adapter=SAMPLER_VARIATION_DEFAULT_ADAPTER,
            c=2,
            beta_decay="logarithmic_decay",
        ),
        LowerBoundSampler(
            interval_width=DEFAULT_INTERVAL_WIDTH,
            adapter=SAMPLER_VARIATION_DEFAULT_ADAPTER,
            c=1,
            beta_decay="logarithmic_decay",
        ),
        PessimisticLowerBoundSampler(
            interval_width=DEFAULT_INTERVAL_WIDTH,
            adapter=SAMPLER_VARIATION_DEFAULT_ADAPTER,
        ),
        ExpectedImprovementSampler(
            n_quantiles=SAMPLER_VARIATION_N_DEFAULT_QUANTILES,
            num_ei_samples=1000,
            adapter=SAMPLER_VARIATION_DEFAULT_ADAPTER,
        ),
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

ARCHITECTURE_VARIATION_ADAPTER = None
ARCHITECTURE_VARIATION_N_QUANTILES = 10
ARCHITECTURE_VARIATION_CONFIGURATIONS = build_architecture_variation_configurations(
    architectures=[
        "qgp",
        "ql",
        "qrf",
        "qgbm",
        "qens3",
        "qens4",
    ],
    samplers=[
        ExpectedImprovementSampler(
            n_quantiles=ARCHITECTURE_VARIATION_N_QUANTILES,
            num_ei_samples=1000,
            adapter=ARCHITECTURE_VARIATION_ADAPTER,
        ),
        ThompsonSampler(
            n_quantiles=ARCHITECTURE_VARIATION_N_QUANTILES,
            enable_optimistic_sampling=False,
            adapter=ARCHITECTURE_VARIATION_ADAPTER,
        ),
        LowerBoundSampler(
            interval_width=DEFAULT_INTERVAL_WIDTH,
            adapter=ARCHITECTURE_VARIATION_ADAPTER,
            c=1,
            beta_decay="logarithmic_decay",
        ),
    ],
)

LIMITED_ARCHITECTURE_ADAPTER = None
LIMITED_ARCHITECTURE_N_QUANTILES = 10
LIMITED_ARCHITECTURE_VARIATION_CONFIGURATIONS = build_architecture_variation_configurations(
    architectures=[
        # "qgp",
        # "qrf",
        "qgbm",
        # "qens3",
        # "qens4",
    ],
    samplers=[
        # ExpectedImprovementSampler(
        #     n_quantiles=LIMITED_ARCHITECTURE_N_QUANTILES,
        #     num_ei_samples=1000,
        #     adapter=LIMITED_ARCHITECTURE_ADAPTER,
        # ),
        ThompsonSampler(
            n_quantiles=LIMITED_ARCHITECTURE_N_QUANTILES,
            enable_optimistic_sampling=False,
            adapter=LIMITED_ARCHITECTURE_ADAPTER,
        ),
    ],
    n_pre_conformal_trials=32,
    searcher_tuning_framework=None,
)

# TODO: TEMP
# LIMITED_ARCHITECTURE_VARIATION_CONFIGURATIONS.extend(build_architecture_variation_configurations(
#     architectures=[
#         # "qrf",
#         # "qgbm",
#     "qgp",
#         # "qens4",
#     ],
#     samplers=[
#         ExpectedImprovementSampler(n_quantiles=50, num_ei_samples=1000, adapter=None),
#         # ThompsonSampler(
#         #     n_quantiles=20, enable_optimistic_sampling=False, adapter=None
#         # ),
#     ],
#     # TODO: TEMP:
#     n_pre_conformal_trials=10000,  # Simulate no pre-conformal trials
# ))


PRECONFORMAL_ADAPTER = None
PRECONFORMAL_N_QUANTILES = 10
PRECONFORMAL_COMPARISON_CONFIGURATIONS = []
for architecture in [
    "qgp",
    "qgbm",
    "qrf",
    "qens3",
    "qens4",
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
                    )
                ],
                n_pre_conformal_trials=pre_conformal_trials,
            )
        )


EXTERNAL_TUNING_CONFIGURATIONS = get_external_tuning_configurations()
