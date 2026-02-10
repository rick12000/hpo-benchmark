try:
    from confopt.selection.acquisition import (
        QuantileConformalSearcher,
    )
    from confopt.selection.sampling.bound_samplers import (
        LowerBoundSampler,
    )
    from confopt.selection.sampling.expected_improvement_samplers import (
        ExpectedImprovementSampler,
    )
    from confopt.selection.sampling.thompson_samplers import ThompsonSampler
except ImportError:
    raise ImportError(
        "confopt is a core dependency of this repository, but it is not automatically installed via pyproject.toml, please refer to the README.md for instructions on how to install this separately"
    )
from hpobench.config.utils import (
    get_external_tuning_configurations,
    build_architecture_variation_configurations,
)
from hpobench.config.config_types import (
    TunerConfig,
    ConfOptModel,
)


# 5. Limited architecture configurations:
LIMITED_ARCHITECTURE_ADAPTER = "DtACI"
LIMITED_ARCHITECTURE_N_QUANTILES = 6
LIMITED_ARCHITECTURE_VARIATION_CONFIGURATIONS = build_architecture_variation_configurations(
    architectures=[
        "qrf",
        "qgbm",
        "ql"
    ],
    samplers=[
        ThompsonSampler(
            n_quantiles=LIMITED_ARCHITECTURE_N_QUANTILES,
            enable_optimistic_sampling=True,
            adapter=LIMITED_ARCHITECTURE_ADAPTER,
        ),
    ],
    n_pre_conformal_trials=32,
    searcher_tuning_framework=None,
    calibration_split_strategy="train_test_split",
)

EXTERNAL_TUNING_CONFIGURATIONS = get_external_tuning_configurations()
