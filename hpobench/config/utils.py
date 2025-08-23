from typing import Union, Optional, List
from confopt.selection.acquisition import (
    QuantileConformalSearcher,
)
from copy import deepcopy
from confopt.selection.sampling.bound_samplers import (
    LowerBoundSampler,
)
from confopt.selection.sampling.entropy_samplers import MaxValueEntropySearchSampler
from confopt.selection.sampling.expected_improvement_samplers import (
    ExpectedImprovementSampler,
)
from confopt.selection.sampling.thompson_samplers import ThompsonSampler
from confopt.selection.acquisition import QuantileEstimatorArchitecture
from hpobench.config.config_types import TunerConfig


def create_sampler_config_id(
    searcher: Union[QuantileConformalSearcher, str],
    custom_prefix: Optional[str] = None,
    searcher_tuning_framework: Optional[str] = None,
) -> str:
    """Create a configuration ID string based on searcher properties.

    Args:
        searcher: A QuantileConformalSearcher instance or a string identifier for the searcher.
        custom_prefix: Optional prefix to add to the config ID.
        searcher_tuning_framework: Optional indicator for tuning framework (e.g., 'fixed', 'reward_cost').

    Returns:
        A formatted string for the configuration ID.
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
            config_id += f"{quantile_arch_upper}-{adapter_name} {sampler_acronym}"
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


def build_static_tuning_configurations(
    quantile_architectures: List[QuantileEstimatorArchitecture],
    searcher_tuning_frameworks: List[Optional[str]],
    n_pre_conformal_trials: int = 20,
) -> List[TunerConfig]:
    """Build static tuning configurations for given quantile architectures and tuning frameworks.

    Args:
        quantile_architectures: List of quantile estimator architectures.
        searcher_tuning_frameworks: List of tuning framework identifiers.
        n_pre_conformal_trials: Number of pre-conformal trials.

    Returns:
        List of static tuning configuration objects.
    """
    # NOTE: Sampler is irrelevant for static tuning analysis,
    # since we only consider the first trial:

    placeholder_sampler = LowerBoundSampler(
        interval_width=0.9,
        adapter="DtACI",
        c=1,
        beta_decay="logarithmic_decay",
    )
    sampler_copy = deepcopy(placeholder_sampler)
    return [
        TunerConfig(
            tuner="confopt",
            searcher=QuantileConformalSearcher(
                quantile_estimator_architecture=arch,
                sampler=sampler_copy,
                n_pre_conformal_trials=n_pre_conformal_trials,
            ),
            config_identifier=create_sampler_config_id(
                QuantileConformalSearcher(
                    quantile_estimator_architecture=arch,
                    sampler=sampler_copy,
                    n_pre_conformal_trials=n_pre_conformal_trials,
                ),
                searcher_tuning_framework=framework,
            ),
            searcher_tuning_framework=framework,
        )
        for arch in quantile_architectures
        for framework in searcher_tuning_frameworks
    ]


def build_sampler_variation_configurations(
    samplers: List[
        Union[
            ThompsonSampler,
            LowerBoundSampler,
            ExpectedImprovementSampler,
            MaxValueEntropySearchSampler,
        ]
    ],
    quantile_arch: QuantileEstimatorArchitecture,
    n_pre_conformal_trials: int = 20,
    searcher_tuning_framework: Optional[str] = None,
) -> List[TunerConfig]:
    """Build tuning configurations for different samplers with a fixed quantile architecture.

    Args:
        samplers: List of sampler instances.
        quantile_arch: Quantile estimator architecture.
        n_pre_conformal_trials: Number of pre-conformal trials.
        searcher_tuning_framework: Value to set in TunerConfig for searcher_tuning_framework.

    Returns:
        List of tuning configuration objects for each sampler.
    """
    configs = []
    for sampler in samplers:
        sampler_copy = deepcopy(sampler)
        searcher = QuantileConformalSearcher(
            quantile_estimator_architecture=quantile_arch,
            sampler=sampler_copy,
            n_pre_conformal_trials=n_pre_conformal_trials,
            n_calibration_folds=5,
            calibration_split_strategy="adaptive",
            symmetric_adjustment=True,
        )
        config_id = create_sampler_config_id(searcher) + (
            f" stf={searcher_tuning_framework}" if searcher_tuning_framework else ""
        )
        configs.append(
            TunerConfig(
                tuner="confopt",
                searcher=searcher,
                config_identifier=config_id,
                searcher_tuning_framework=searcher_tuning_framework,
            )
        )
    return configs


def build_architecture_variation_configurations(
    architectures: List[QuantileEstimatorArchitecture],
    samplers: List[
        Union[
            ThompsonSampler,
            LowerBoundSampler,
            ExpectedImprovementSampler,
            MaxValueEntropySearchSampler,
        ]
    ],
    n_pre_conformal_trials: int = 20,
    searcher_tuning_framework: Optional[str] = None,
) -> List[TunerConfig]:
    """Build tuning configurations for different quantile architectures and samplers.

    Args:
        architectures: List of quantile estimator architectures.
        samplers: List of sampler instances.
        n_pre_conformal_trials: Number of pre-conformal trials.
        searcher_tuning_framework: Value to set in TunerConfig for searcher_tuning_framework.

    Returns:
        List of tuning configuration objects for each architecture and sampler combination.
    """
    configs = []
    for arch in architectures:
        for sampler in samplers:
            sampler_copy = deepcopy(sampler)
            searcher = QuantileConformalSearcher(
                quantile_estimator_architecture=arch,
                sampler=sampler_copy,
                n_pre_conformal_trials=n_pre_conformal_trials,
                n_calibration_folds=5,
                calibration_split_strategy="adaptive",
                symmetric_adjustment=True,
            )
            config_id = create_sampler_config_id(searcher) + (
                f" stf={searcher_tuning_framework}" if searcher_tuning_framework else ""
            )
            configs.append(
                TunerConfig(
                    tuner="confopt",
                    searcher=searcher,
                    config_identifier=config_id,
                    searcher_tuning_framework=searcher_tuning_framework,
                )
            )
    return configs


def get_external_tuning_configurations() -> List[TunerConfig]:
    """Get external (non-confopt) tuning configurations for baseline comparison.

    Returns:
        List of external tuning configuration objects (e.g., for skopt, optuna).
    """
    return [
        # TunerConfig(
        #     tuner="skopt",
        #     searcher="gp",
        #     config_identifier="GP1",
        # ),
        TunerConfig(
            tuner="optuna",
            searcher="tpe",
            config_identifier="TPE",
        ),
        # TunerConfig(
        #     tuner="optuna",
        #     searcher="gp",
        #     config_identifier="GP2",
        # ),
        # Syne-Tune CQR configurations using string searchers
        # TunerConfig(
        #     tuner="syne_tune_cqr",
        #     searcher="cqr_thompson",
        #     config_identifier="CQR-THOMPSON",
        # ),
        TunerConfig(
            tuner="optuna",
            searcher="random",
            config_identifier="RS",
        ),
        # TunerConfig(
        #     tuner="skopt",
        #     searcher="gbrt",
        #     config_identifier="GBRT",
        # ),
        TunerConfig(
            tuner="optuna",
            searcher="confopt_gp_expected_improvement",
            config_identifier="GP-EI",
        ),
        TunerConfig(
            tuner="optuna",
            searcher="confopt_gp_thompson_sampling",
            config_identifier="GP-TS",
        ),
        # TunerConfig(
        #     tuner="smac",
        #     searcher="smac_rf_ei",
        #     config_identifier="RF-EI",
        # )
    ]
