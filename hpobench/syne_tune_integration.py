"""
Integration module for Syne-Tune's Conformal Quantile Regression (CQR) searcher
into the HPO benchmark framework.

This module adapts Syne-Tune's CQR implementation to work with the standardized
tuning interface, allowing fair comparison with other HPO methods.
"""

import pandas as pd
from datetime import datetime
from typing import Union, Optional, Any, Dict, List, Tuple
import logging

from syne_tune.config_space import Domain, Float, Integer, Categorical
from syne_tune.optimizer.schedulers.searchers.conformal.surrogate.surrogate_model import (
    SurrogateModel,
)
from syne_tune.optimizer.schedulers.searchers.conformal.surrogate.quantile_regression_surrogate import (
    QuantileRegressionSurrogateModel,
)

from hpobench.config.types import IntRange, FloatRange, CategoricalRange
from hpobench.generation.generate import ObjectiveMetricGenerator

logger = logging.getLogger(__name__)


def build_history_entry(
    end_time: Optional[Any] = None,
    performance: Optional[Any] = None,
    configurations: Optional[Any] = None,
    iteration: Optional[int] = None,
    estimator_error: Optional[Any] = None,
    searcher_training_time: Optional[Any] = None,
    breach_status: Optional[int] = None,
    winkler_score: Optional[float] = None,
    width: Optional[float] = None,
    miscoverage_penalty: Optional[float] = None,
    tabularized_configuration: Optional[Any] = None,
) -> dict[str, Any]:
    """Standardizes the history entry structure for syne-tune integration."""
    return {
        "end_time": end_time,
        "performance": performance,
        "configurations": configurations,
        "iteration": iteration,
        "estimator_error": estimator_error,
        "searcher_training_time": searcher_training_time,
        "breach_status": breach_status,
        "winkler_score": winkler_score,
        "width": width,
        "miscoverage_penalty": miscoverage_penalty,
        "tabularized_configuration": tabularized_configuration,
    }


# Constants for CQR configuration
DEFAULT_NUM_INIT_RANDOM_DRAWS = 5
DEFAULT_UPDATE_FREQUENCY = 1
DEFAULT_MAX_FIT_SAMPLES = 1000
DEFAULT_QUANTILES = 5
DEFAULT_MIN_SAMPLES_TO_CONFORMALIZE = 32
DEFAULT_VALID_FRACTION = 0.1


def _create_cqr_params(
    acquisition_strategy: str, num_warm_starts: int = 0
) -> dict[str, Any]:
    """Create CQR parameters dictionary with default values and specified acquisition strategy.

    Args:
        acquisition_strategy: The acquisition strategy to use.
        num_warm_starts: Number of warm start configurations. If > 0, num_init_random_draws is set to 0.

    Returns:
        Dictionary with CQR configuration parameters.
    """
    # If we have warm starts, don't do additional random draws since warm starts count towards init draws
    num_init_random_draws = 0 if num_warm_starts > 0 else DEFAULT_NUM_INIT_RANDOM_DRAWS

    return {
        "num_init_random_draws": num_init_random_draws,
        "update_frequency": DEFAULT_UPDATE_FREQUENCY,
        "max_fit_samples": DEFAULT_MAX_FIT_SAMPLES,
        "quantiles": DEFAULT_QUANTILES,
        "min_samples_to_conformalize": DEFAULT_MIN_SAMPLES_TO_CONFORMALIZE,
        "valid_fraction": DEFAULT_VALID_FRACTION,
        "acquisition_strategy": acquisition_strategy,
    }


def convert_params_to_syne_tune_config_space(
    raw_params: dict[str, Union[IntRange, FloatRange, CategoricalRange]]
) -> dict[str, Domain]:
    """
    Converts benchmarking framework parameter types to Syne-Tune Domain objects.

    Args:
        raw_params: Dictionary mapping parameter names to IntRange, FloatRange, or CategoricalRange.

    Returns:
        Dictionary mapping parameter names to Syne-Tune Domain objects.
    """

    config_space = {}

    for name, param in raw_params.items():
        if isinstance(param, IntRange):
            config_space[name] = Integer(lower=param.lower, upper=param.upper)
        elif isinstance(param, FloatRange):
            config_space[name] = Float(lower=param.lower, upper=param.upper)
        elif isinstance(param, CategoricalRange):
            config_space[name] = Categorical(categories=param.choices)
        else:
            raise ValueError(f"Unknown parameter type: {type(param)}")

    return config_space


class CustomSurrogateSearcher(SurrogateModel):
    """Custom SurrogateModel that allows full control over surrogate model parameters."""

    def __init__(self, *args, **kwargs):
        # Extract our custom parameters
        self.custom_min_samples_to_conformalize = kwargs.pop(
            "min_samples_to_conformalize", 32
        )
        self.custom_valid_fraction = kwargs.pop("valid_fraction", 0.1)
        self.custom_quantiles = kwargs.pop("quantiles", 5)

        super().__init__(*args, **kwargs)

    def fit_model(self):
        """Override fit_model to use our custom parameters."""
        X, z = self.make_input_target()

        # Use our custom parameters instead of hardcoded ones
        self.surrogate_model = QuantileRegressionSurrogateModel(
            config_space=self.config_space,
            max_fit_samples=self.max_fit_samples,
            random_state=self.random_state,
            mode="min",
            min_samples_to_conformalize=self.custom_min_samples_to_conformalize,
            valid_fraction=self.custom_valid_fraction,
            quantiles=self.custom_quantiles,
            **{
                k: v
                for k, v in self.surrogate_kwargs.items()
                if k
                not in ["min_samples_to_conformalize", "valid_fraction", "quantiles"]
            },
        )
        self.surrogate_model.fit(df_features=X, y=z)


class SyneTuneCQRWrapper:
    """
    Wrapper class that adapts Syne-Tune's CQR searcher to the benchmarking framework interface.

    This class takes CQR configuration parameters and instantiates the actual
    CustomSurrogateSearcher that implements the CQR functionality.
    """

    def __init__(
        self,
        raw_params: dict[str, Union[IntRange, FloatRange, CategoricalRange]],
        performance_generator: ObjectiveMetricGenerator,
        cqr_params: dict[str, Any],
        warm_start_configs: Optional[List[Tuple[dict, float]]] = None,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize the Syne-Tune CQR wrapper.

        Args:
            raw_params: Parameter space definition.
            performance_generator: Objective function generator.
            cqr_params: Dictionary containing CQR-specific parameters.
            warm_start_configs: Optional warm start configurations.
            random_seed: Random seed for reproducibility.
        """

        self.raw_params = raw_params
        self.performance_generator = performance_generator
        self.random_seed = random_seed
        self.cqr_params = cqr_params

        # Convert parameter space to Syne-Tune format
        self.syne_tune_config_space = convert_params_to_syne_tune_config_space(
            raw_params
        )

        # Convert warm start configurations
        warm_start_points = None
        warm_start_results = None
        if warm_start_configs:
            warm_start_points = [config for config, _ in warm_start_configs]
            warm_start_results = [loss for _, loss in warm_start_configs]

        # Initialize the custom Syne-Tune CQR searcher using parameters from cqr_params
        # Note: num_init_random_draws is set to 0 when warm_start_configs are provided
        # because the warm start configs (via points_to_evaluate) count towards the
        # initial observations needed before the surrogate model is used
        self.searcher = CustomSurrogateSearcher(
            config_space=self.syne_tune_config_space,
            num_init_random_draws=cqr_params["num_init_random_draws"],
            update_frequency=cqr_params["update_frequency"],
            points_to_evaluate=warm_start_points,
            max_fit_samples=cqr_params["max_fit_samples"],
            random_seed=random_seed,
            quantiles=cqr_params["quantiles"],
            min_samples_to_conformalize=cqr_params["min_samples_to_conformalize"],
            valid_fraction=cqr_params["valid_fraction"],
        )

        # Store acquisition strategy for later use
        self.acquisition_strategy = cqr_params["acquisition_strategy"]

        # Track evaluation history
        self.history = []
        self.trial_counter = 0

        # If we have warm start configs, we need to report them to the searcher
        # when they are evaluated, using the known results
        self.warm_start_configs = warm_start_configs or []
        self.warm_start_results = warm_start_results or []

    def suggest_configuration(self) -> Dict[str, Any]:
        """
        Suggest the next configuration to evaluate.

        Returns:
            Dictionary with parameter configuration.
        """
        # Get suggestion from Syne-Tune searcher
        # The searcher will handle warm starts through points_to_evaluate internally
        syne_tune_config = self.searcher.suggest()

        if syne_tune_config is None:
            raise RuntimeError("Searcher failed to suggest a configuration")

        return syne_tune_config

    def report_result(self, config: Dict[str, Any], performance: float) -> None:
        """
        Report the result of evaluating a configuration.

        Args:
            config: Configuration that was evaluated.
            performance: Performance metric value.
        """
        # Check if this is a warm start configuration and use the known result
        for i, (warm_config, warm_perf) in enumerate(self.warm_start_configs):
            if config == warm_config:
                # Use the known performance from warm start
                performance = warm_perf
                break

        # Report to the searcher
        self.searcher.on_trial_complete(
            trial_id=self.trial_counter, config=config, metric=performance
        )

        # Add to our history
        self.history.append(
            build_history_entry(
                end_time=datetime.now(),
                performance=performance,
                configurations=config,
                iteration=self.trial_counter + 1,
                estimator_error=None,
                searcher_training_time=None,
                breach_status=None,
                winkler_score=None,
                width=None,
                miscoverage_penalty=None,
                tabularized_configuration=None,
            )
        )
        self.trial_counter += 1

    def get_history_dataframe(self) -> pd.DataFrame:
        """
        Get the optimization history as a DataFrame.

        Returns:
            DataFrame with optimization history.
        """
        return pd.DataFrame(self.history)


def syne_tune_cqr_tune(
    raw_params: dict[str, Union[IntRange, FloatRange, CategoricalRange]],
    performance_generator: ObjectiveMetricGenerator,
    sampler: str,
    warm_start_configs: Optional[List[Tuple[dict, float]]] = None,
    random_state: Optional[int] = None,
    n_trials: Optional[int] = None,
    timeout: Optional[float] = None,
) -> pd.DataFrame:
    """
    Runs Syne-Tune CQR tuning with a synthetic objective.

    Args:
        raw_params: Dictionary mapping parameter names to IntRange, FloatRange, or CategoricalRange.
        performance_generator: ObjectiveMetricGenerator instance.
        sampler: String identifier for the CQR acquisition strategy.
        warm_start_configs: Optional list of (config, loss) tuples for warm start.
        random_state: Optional random seed.
        n_trials: Optional number of trials.
        timeout: Optional time budget in seconds.

    Returns:
        DataFrame with tuning history.
    """

    # Map string sampler to CQR configuration parameters
    acquisition_strategy_map = {
        "cqr_thompson": "thompson",
        "cqr_ucb": "ucb",
        "cqr_optimistic": "optimistic",
        "cqr_pessimistic": "pessimistic",
    }

    if sampler not in acquisition_strategy_map:
        raise ValueError(f"Unknown Syne-Tune CQR sampler: {sampler}")

    # Calculate number of warm starts
    num_warm_starts = len(warm_start_configs) if warm_start_configs else 0

    cqr_params = _create_cqr_params(acquisition_strategy_map[sampler], num_warm_starts)

    logger.info(
        f"Syne-Tune CQR configuration: {sampler} with {num_warm_starts} warm starts, "
        f"num_init_random_draws={cqr_params['num_init_random_draws']}"
    )

    # Initialize the wrapper with the configuration parameters
    wrapper = SyneTuneCQRWrapper(
        raw_params=raw_params,
        performance_generator=performance_generator,
        cqr_params=cqr_params,
        warm_start_configs=warm_start_configs,
        random_seed=random_state,
    )

    # Calculate number of trials to run
    if n_trials is not None:
        # Since warm start configs are handled internally by Syne-Tune through points_to_evaluate,
        # we don't need to subtract them from the total number of trials
        adj_n_trials = n_trials
    else:
        adj_n_trials = 100  # Default number of trials

    # Main optimization loop
    start_time = datetime.now()

    for trial_idx in range(adj_n_trials):
        # Check timeout
        if timeout is not None:
            elapsed_time = (datetime.now() - start_time).total_seconds()
            if elapsed_time >= timeout:
                logger.info(f"Timeout reached after {elapsed_time:.2f} seconds")
                break

        try:
            # Get next configuration
            config = wrapper.suggest_configuration()

            # Evaluate configuration
            performance = performance_generator.predict(configuration=config)

            # Report result
            wrapper.report_result(config, performance)

            logger.debug(f"Trial {trial_idx + 1}: {config} -> {performance}")

        except Exception as e:
            logger.error(f"Error in trial {trial_idx + 1}: {e}")
            break

    return wrapper.get_history_dataframe()
