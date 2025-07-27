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
from syne_tune.optimizer.schedulers.searchers.conformal.conformal_quantile_regression_searcher import (
    ConformalQuantileRegression,
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


def _create_cqr_params(num_warm_starts: int = 0) -> dict[str, Any]:
    """Create CQR parameters dictionary with default values.

    Args:
        num_warm_starts: Number of warm start configurations.

    Returns:
        Dictionary with CQR configuration parameters.
    """
    # Use a reasonable number of initial random draws
    # The CQR model needs sufficient data to fit properly
    if num_warm_starts >= DEFAULT_NUM_INIT_RANDOM_DRAWS:
        # If we have enough warm starts, use fewer additional random draws
        num_init_random_draws = 2  # Still do a few to ensure model stability
    else:
        # If we have few or no warm starts, use the default
        num_init_random_draws = DEFAULT_NUM_INIT_RANDOM_DRAWS

    return {
        "num_init_random_draws": num_init_random_draws,
        "update_frequency": DEFAULT_UPDATE_FREQUENCY,
        "max_fit_samples": DEFAULT_MAX_FIT_SAMPLES,
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


class FixedConformalQuantileRegression(ConformalQuantileRegression):
    """
    Fixed version of ConformalQuantileRegression that avoids parameter conflicts
    in the fit_model method.
    """

    def fit_model(self):
        """Override fit_model to avoid parameter conflicts."""
        X, z = self.make_input_target()

        # Filter out conflicting parameters from surrogate_kwargs
        safe_surrogate_kwargs = {
            k: v
            for k, v in self.surrogate_kwargs.items()
            if k not in ["min_samples_to_conformalize", "valid_fraction"]
        }

        logger.debug(f"Fitting CQR model with {len(X)} samples")
        logger.debug(f"Filtered surrogate_kwargs: {safe_surrogate_kwargs}")

        try:
            self.surrogate_model = self.surrogate_cls(
                config_space=self.config_space,
                max_fit_samples=self.max_fit_samples,
                random_state=self.random_state,
                mode="min",
                min_samples_to_conformalize=32,
                valid_fraction=0.1,
                **safe_surrogate_kwargs,
            )
            self.surrogate_model.fit(df_features=X, y=z)
            logger.debug("CQR model fitted successfully")
        except Exception as e:
            logger.error(f"Failed to fit CQR model: {e}")
            raise


class SyneTuneCQRWrapper:
    """
    Wrapper class that adapts Syne-Tune's CQR searcher to the benchmarking framework interface.

    This class takes CQR configuration parameters and instantiates the actual
    ConformalQuantileRegression searcher that implements the CQR functionality.
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
        if warm_start_configs:
            warm_start_points = [config for config, _ in warm_start_configs]

        # Initialize the Syne-Tune CQR searcher using our fixed version
        # that avoids parameter conflicts in fit_model
        self.searcher = FixedConformalQuantileRegression(
            config_space=self.syne_tune_config_space,
            points_to_evaluate=warm_start_points,
            num_init_random_draws=cqr_params["num_init_random_draws"],
            update_frequency=cqr_params["update_frequency"],
            max_fit_samples=cqr_params["max_fit_samples"],
            random_seed=random_seed,
            # Do not pass any surrogate_kwargs to avoid conflicts with hardcoded parameters
            # in fit_model (min_samples_to_conformalize=32, valid_fraction=0.1)
        )

        # Track evaluation history
        self.history = []
        self.trial_counter = 0

        # If we have warm start configs, we need to report them to the searcher
        # when they are evaluated, using the known results
        self.warm_start_configs = warm_start_configs or []

    def suggest_configuration(self) -> Dict[str, Any]:
        """
        Suggest the next configuration to evaluate.

        Returns:
            Dictionary with parameter configuration.
        """
        logger.debug(f"Requesting suggestion for trial {self.trial_counter}")
        logger.debug(f"Searcher has {self.searcher.num_results()} results")
        logger.debug(f"Should update model: {self.searcher.should_update()}")

        try:
            # Get suggestion from Syne-Tune searcher
            # The searcher will handle warm starts through points_to_evaluate internally
            syne_tune_config = self.searcher.suggest()

            if syne_tune_config is None:
                # This should not happen in normal operation, but if it does,
                # fall back to random sampling to ensure we can continue
                logger.warning(
                    "Searcher returned None, falling back to random sampling"
                )
                syne_tune_config = self.searcher.sample_random()

        except Exception as e:
            # If the searcher fails (e.g., model fitting error), fall back to random sampling
            logger.error(
                f"Searcher failed with error: {e}, falling back to random sampling"
            )
            syne_tune_config = self.searcher.sample_random()

        logger.debug(
            f"Suggested config for trial {self.trial_counter}: {syne_tune_config}"
        )
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
                logger.debug(
                    f"Using warm start performance for trial {self.trial_counter}: {performance}"
                )
                break

        # Report to the searcher
        self.searcher.on_trial_complete(
            trial_id=self.trial_counter, config=config, metric=performance
        )

        logger.debug(
            f"Reported trial {self.trial_counter}: config={config}, performance={performance}"
        )
        logger.debug(f"Searcher now has {self.searcher.num_results()} results")

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
        sampler: String identifier for the CQR sampler (only "cqr_thompson" is supported since CQR uses Thompson sampling).
        warm_start_configs: Optional list of (config, loss) tuples for warm start.
        random_state: Optional random seed.
        n_trials: Optional number of trials.
        timeout: Optional time budget in seconds.

    Returns:
        DataFrame with tuning history.
    """

    # Syne-Tune CQR only supports Thompson sampling (random sampling from quantiles)
    # All the different "acquisition strategies" map to the same implementation
    supported_samplers = {
        "cqr_thompson",
        "cqr_ucb",
        "cqr_optimistic",
        "cqr_pessimistic",
    }

    if sampler not in supported_samplers:
        raise ValueError(
            f"Unknown Syne-Tune CQR sampler: {sampler}. Supported: {supported_samplers}"
        )

    # Calculate number of warm starts
    num_warm_starts = len(warm_start_configs) if warm_start_configs else 0

    cqr_params = _create_cqr_params(num_warm_starts)

    logger.info(
        f"Syne-Tune CQR configuration: {sampler} with {num_warm_starts} warm starts, "
        f"num_init_random_draws={cqr_params['num_init_random_draws']} "
        f"(Note: All CQR samplers use Thompson sampling internally)"
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
        # Follow the same pattern as other tuners: subtract warm start configs from total
        # since warm start configs count as trials but are handled through points_to_evaluate
        if warm_start_configs is not None:
            adj_n_trials = n_trials  # - len(warm_start_configs)
        else:
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
            # Don't break - log the error and continue with the next trial
            # This ensures we complete the requested number of trials even if some fail
            continue

    return wrapper.get_history_dataframe()
