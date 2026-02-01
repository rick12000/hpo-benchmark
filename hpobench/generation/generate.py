import numpy as np
from typing import Union, Dict, Any, Literal
from pathlib import Path

from ConfigSpace import Configuration

from abc import ABC, abstractmethod

from yahpo_gym import BenchmarkSet

from hpobench.generation.black_box_functions import (
    rastrigin,
    ackley,
    griewank,
    weierstrass,
    shekel,
    hartmann6,
)
from hpobench.config.constants import SYNTHETIC_TABULAR_STORAGE_DIR
from hpobench.generation.tabular.storage import DatasetStorage

import logging
logger = logging.getLogger(__name__)

def _ensure_yahpo_initialized():
    """Wrapper to avoid circular imports."""
    from hpobench.utils import ensure_yahpo_initialized

    ensure_yahpo_initialized()


class ObjectiveMetricGenerator(ABC):
    """Abstract base class for objective metric generators used in benchmarking.

    Args:
        generator: Name or identifier for the generator (used by subclasses).
    """

    def __init__(self, generator: str):
        self.generator = generator

    @abstractmethod
    def predict(self, configuration: dict[str, Union[str, int, float, bool]]) -> float:
        """Return the objective value for a given configuration.

        Args:
            configuration: Dictionary mapping parameter names to their values.

        Returns:
            The objective value for the given configuration.
        """

    @abstractmethod
    def predict_batch(self, configurations: list[dict]) -> list[float]:
        """Return objective values for multiple configurations in batch.

        Args:
            configurations: List of configuration dictionaries to evaluate.

        Returns:
            List of objective values for the given configurations.
        """

    @abstractmethod
    def predict_runtime(
        self, configuration: dict[str, Union[str, int, float, bool]]
    ) -> float:
        """Return the runtime for a given configuration.

        Args:
            configuration: Dictionary mapping parameter names to their values.

        Returns:
            The runtime for the given configuration.
        """

    @abstractmethod
    def predict_runtime_batch(self, configurations: list[dict]) -> list[float]:
        """Return runtime values for multiple configurations in batch.

        Args:
            configurations: List of configuration dictionaries to evaluate.

        Returns:
            List of runtime values for the given configurations.
        """

    def initialize(self) -> None:
        """Initialize the generator if needed. Default implementation does nothing.

        Returns:
            None
        """


class BlackBoxGenerator(ObjectiveMetricGenerator):
    """Objective metric generator for standard black-box optimization functions.

    Args:
        generator: Name of the black-box function to use (e.g., 'rastrigin', 'ackley').
    """

    def __init__(self, generator: str):
        self.generator = generator

    def _evaluate_function(self, x: np.ndarray) -> float:
        """Evaluate the black-box function for the given parameter vector.

        Args:
            x: Parameter vector as numpy array.

        Returns:
            Function value.
        """
        if self.generator == "rastrigin":
            return rastrigin(x=x)
        elif self.generator == "ackley":
            return ackley(x=x)
        elif self.generator == "griewank":
            return griewank(x=x)
        elif self.generator == "weierstrass":
            return weierstrass(x=x)
        elif self.generator == "shekel":
            return shekel(x=x)
        elif self.generator == "hartmann6":
            return hartmann6(x=x)
        else:
            raise ValueError(f"Unknown generator: {self.generator}")

    def predict(self, configuration: dict[str, Union[str, int, float, bool]]) -> float:
        """Evaluate the black-box function for the given configuration.

        Args:
            configuration: Dictionary mapping parameter names to their values.

        Returns:
            The function value for the given configuration.
        """
        x = np.array(list(configuration.values()), dtype=float)
        return self._evaluate_function(x)

    def predict_batch(self, configurations: list[dict]) -> list[float]:
        """Evaluate multiple configurations in batch.

        Args:
            configurations: List of configuration dictionaries to evaluate.

        Returns:
            List of function values.
        """
        return [self.predict(config) for config in configurations]

    def predict_runtime(
        self, configuration: dict[str, Union[str, int, float, bool]]
    ) -> float:
        """Return the runtime for the given configuration (always 0 for black-box functions).

        Args:
            configuration: Dictionary mapping parameter names to their values.

        Returns:
            Always returns 0.
        """
        return 0

    def predict_runtime_batch(self, configurations: list[dict]) -> list[float]:
        """Return runtime for multiple configurations (always 0 for black-box functions).

        Args:
            configurations: List of configuration dictionaries.

        Returns:
            List of runtime values (all 0 for black-box functions).
        """
        return [0.0] * len(configurations)


class YahpoGenerator(ObjectiveMetricGenerator):
    """Objective metric generator for YAHPO Gym surrogate benchmarks.

    Handles instance-specific and fidelity-aware configuration evaluation.
    Automatically uses maximum fidelity values for all fidelity parameters.
    
    Tracks all queries to build surrogate data for metafeature calculation.

    Args:
        dataset: Name of the YAHPO benchmark scenario.
        instance_value: Value of the instance for this experiment.
        instance_name: Name of the instance parameter in the configuration space.
        fidelity_space: Dictionary of fidelity parameter names and their MAXIMUM values.
        config_space: Configuration space object.
    """

    def __init__(
        self,
        dataset: str,
        instance_value: Any,
        instance_name: str,
        fidelity_space: Dict,
        config_space,
    ):
        # Ensure YAHPO is initialized before creating BenchmarkSet
        _ensure_yahpo_initialized()

        self.dataset = dataset
        self.instance_name = instance_name
        self.generator = BenchmarkSet(
            dataset, instance=instance_value, active_session=False
        )

        self.config_space = config_space
        # Store maximum fidelity values (passed from setup functions)
        self.fidelity_space = fidelity_space

    def _get_filtered_configuration(self, configuration: dict) -> dict:
        """Filter the configuration to include only active and fidelity parameters.

        Uses ConfigSpace's built-in get_active_hyperparameters method for robust
        conditional dependency handling.

        Args:
            configuration: Dictionary mapping parameter names to their values.

        Returns:
            Filtered configuration dictionary including only active and fidelity parameters.
        """
        config_dict = configuration.copy()

        # Add fidelity parameters to the configuration for evaluation
        if self.fidelity_space:
            config_dict.update(self.fidelity_space)

        # Add instance parameter for evaluation
        config_dict[self.instance_name] = self.generator.instance

        # Use ConfigSpace's built-in method to get active hyperparameters
        try:
            cs_config = Configuration(
                self.config_space,
                values=config_dict,
                allow_inactive_with_values=True,
            )
            active_hyperparameters = self.config_space.get_active_hyperparameters(
                cs_config
            )

            # Filter configuration to only include active parameters
            filtered_configuration = {
                k: v
                for k, v in config_dict.items()
                if k in active_hyperparameters or k == self.instance_name
            }

        except Exception as e:
            raise ValueError(f"ConfigSpace evaluation failed: {e}")

        return filtered_configuration

    def predict(self, configuration: dict[str, Union[str, int, float, bool]]) -> float:
        """Return the negative primary metric for the given configuration.

        Args:
            configuration: Dictionary mapping parameter names to their values.

        Returns:
            Negative primary metric.
        """
        batch_results = self._batch_evaluate_configurations([configuration])
        performance = self._extract_performance_metric(batch_results[0])
        return performance

    def _batch_evaluate_configurations(self, configurations: list[dict]) -> list[dict]:
        """Helper method to filter and evaluate multiple configurations in batch.

        Args:
            configurations: List of configuration dictionaries to evaluate.

        Returns:
            List of batch evaluation results.
        """
        filtered_configs = []
        for config in configurations:
            filtered_config = self._get_filtered_configuration(config)
            filtered_configs.append(filtered_config)

        return self.generator.objective_function(filtered_configs, seed=1234)

    def _extract_performance_metric(self, result: dict) -> float:
        """Extract performance metric from evaluation result.

        Args:
            result: Single evaluation result dictionary.

        Returns:
            Performance value (negated for minimization).
        """
        if "val_accuracy" in result:
            return -result["val_accuracy"]
        elif "acc" in result:
            return -result["acc"]
        elif "auc" in result:
            return -result["auc"]
        else:
            raise ValueError(
                f"No suitable metric found in results: {list(result.keys())}"
            )

    def _extract_runtime_metric(self, result: dict) -> float:
        """Extract runtime metric from evaluation result.

        Args:
            result: Single evaluation result dictionary.

        Returns:
            Runtime value.
        """
        if "time" in result:
            return result["time"]
        elif "runtime" in result:
            return result["runtime"]
        else:
            return result["timetrain"] + result["timepredict"]

    def predict_batch(self, configurations: list[dict]) -> list[float]:
        """Evaluate multiple configurations in batch for improved performance.
        
        Also tracks queries in history for surrogate metafeature calculation.

        Args:
            configurations: List of configuration dictionaries to evaluate.

        Returns:
            List of performance values (negated for minimization).
        """
        batch_results = self._batch_evaluate_configurations(configurations)
        performances = [self._extract_performance_metric(result) for result in batch_results]
        return performances

    def predict_runtime(
        self, configuration: dict[str, Union[str, int, float, bool]]
    ) -> float:
        """Return the runtime for the given configuration.

        Args:
            configuration: Dictionary mapping parameter names to their values.

        Returns:
            Runtime.
        """
        batch_results = self._batch_evaluate_configurations([configuration])
        return self._extract_runtime_metric(batch_results[0])

    def predict_runtime_batch(self, configurations: list[dict]) -> list[float]:
        """Evaluate runtime for multiple configurations in batch.

        Args:
            configurations: List of configuration dictionaries to evaluate.

        Returns:
            List of runtime values.
        """
        batch_results = self._batch_evaluate_configurations(configurations)
        return [self._extract_runtime_metric(result) for result in batch_results]


class SyntheticTabularGenerator(ObjectiveMetricGenerator):
    """Generator for synthetic surrogate data.
    
    The SCM-generated synthetic data represents surrogate performance landscapes
    (hyperparameter configurations and their performances). The features (X) represent
    hyperparameter configurations, and the targets (y) represent performance values.
    """
    
    def __init__(
        self,
        generator: str,
        dataset: str,
        model_type: Literal["random_forest", "gradient_boosted_trees"],
        train_size: float = 0.8,
        random_state: int = 42,
    ):
        self.generator = generator
        self.dataset = dataset
        self.storage_dir = Path(SYNTHETIC_TABULAR_STORAGE_DIR)
        self.model_type = model_type  # Kept for compatibility but not used
        self.train_size = train_size  # Kept for compatibility but not used
        self.random_state = random_state
        
        # Surrogate data (X = configs, y = performances)
        self.surrogate_features = None  # Hyperparameter configurations
        self.surrogate_targets = None  # Performance values
        self.dataset_metadata = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Load SCM-generated surrogate data.
        
        The SCM generator creates synthetic data where:
        - Features (X) = hyperparameter configurations
        - Targets (y) = performance values
        """
        if self._initialized:
            return

        try:
            storage = DatasetStorage(str(self.storage_dir))
            dataset_id = int(self.dataset)
            
            # Load dataset using existing method
            # X represents hyperparameter configs, y represents performances
            self.surrogate_features, self.surrogate_targets, self.dataset_metadata = (
                storage.load_dataset(dataset_id)
            )
            logger.info(
                f"Loaded surrogate dataset {dataset_id} with "
                f"{len(self.surrogate_features)} configurations"
            )
        except Exception as e:
            logger.error(f"Failed to load dataset {self.dataset}: {e}", exc_info=True)
            raise

        self._initialized = True
    
    def predict(self, configuration: dict[str, Union[str, int, float, bool]]) -> float:
        """Predict performance by looking up in surrogate data.
        
        The surrogate data (X, y) represents (configs, performances).
        We find the nearest configuration and return its performance.
        """
        self.initialize()
        
        # Convert configuration to feature vector
        config_vec = self._config_to_vector(configuration)
        
        # Find nearest neighbor in surrogate data
        X = self.surrogate_features.values
        y = self.surrogate_targets.values.ravel()
        
        # Calculate distances
        distances = np.linalg.norm(X - config_vec, axis=1)
        nearest_idx = np.argmin(distances)
        
        return float(y[nearest_idx])
    
    def _config_to_vector(self, configuration: dict) -> np.ndarray:
        """Convert configuration dict to feature vector matching surrogate data format."""
        # Get feature names from surrogate data
        feature_names = self.surrogate_features.columns.tolist()
        
        # Create vector in same order as features
        vec = []
        for feature_name in feature_names:
            if feature_name in configuration:
                val = configuration[feature_name]
                if isinstance(val, (int, float)):
                    vec.append(float(val))
                else:
                    # For categorical, try to convert or use hash
                    try:
                        vec.append(float(val))
                    except:
                        vec.append(float(hash(str(val)) % 1000))
            else:
                # Feature not in config, use 0
                vec.append(0.0)
        
        return np.array(vec).reshape(1, -1)
    
    def predict_batch(self, configurations: list[dict]) -> list[float]:
        """Batch prediction using surrogate data lookup."""
        results = []
        for config in configurations:
            results.append(self.predict(config))
        return results
    
    def predict_runtime(
        self, configuration: dict[str, Union[str, int, float, bool]]
    ) -> float:
        return 0.0
    
    def predict_runtime_batch(self, configurations: list[dict]) -> list[float]:
        return [0.0] * len(configurations)
