import numpy as np
from typing import Union, Dict, Any, Optional
from pathlib import Path

from ConfigSpace import Configuration

from yahpo_gym import BenchmarkSet

from hpobench.generation.base import ObjectiveMetricGenerator
from hpobench.generation.black_box import (
    rastrigin,
    ackley,
    griewank,
    weierstrass,
    shekel,
    hartmann6,
)
from hpobench.config.constants import SyntheticGenerationParameters
from hpobench.generation.tabular.storage import DatasetStorage

import logging
logger = logging.getLogger(__name__)

def _ensure_yahpo_initialized():
    """Wrapper to avoid circular imports."""
    from hpobench.utils import ensure_yahpo_initialized

    ensure_yahpo_initialized()


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
            return rastrigin(input_vector=x)
        elif self.generator == "ackley":
            return ackley(input_vector=x)
        elif self.generator == "griewank":
            return griewank(input_vector=x)
        elif self.generator == "weierstrass":
            return weierstrass(input_vector=x)
        elif self.generator == "shekel":
            return shekel(input_vector=x)
        elif self.generator == "hartmann6":
            return hartmann6(input_vector=x)
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

    def _filter_and_enhance_configuration(self, configuration: dict) -> dict:
        """Filter configuration to active parameters and add fidelity/instance parameters.

        Uses ConfigSpace's built-in get_active_hyperparameters method for robust
        conditional dependency handling. Adds fidelity parameters and instance
        parameter required for benchmark evaluation.

        Args:
            configuration: Dictionary mapping parameter names to their values.

        Returns:
            Filtered configuration dictionary with only active parameters plus
            fidelity and instance parameters.
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
            filtered_config = {
                k: v
                for k, v in config_dict.items()
                if k in active_hyperparameters or k == self.instance_name
            }

        except Exception as e:
            raise ValueError(f"ConfigSpace evaluation failed: {e}")

        return filtered_config

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
            filtered_config = self._filter_and_enhance_configuration(config)
            filtered_configs.append(filtered_config)

        return self.generator.objective_function(filtered_configs, seed=1234)

    def _extract_performance_metric(self, result: dict) -> float:
        """Extract and negate performance metric from evaluation result.
        
        Looks for standard performance metrics (accuracy, AUC) and returns 
        the negated value for minimization-based HPO.

        Args:
            result: Single evaluation result dictionary.

        Returns:
            Negated performance value (for minimization).
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


class SyntheticGenerator(ObjectiveMetricGenerator):
    """Generator for synthetic HPO performance landscape data.

    Provides access to precomputed synthetic surrogate performance landscapes
    (hyperparameter configurations and their mean-loss values), similar to
    YAHPO/lcbench.  The features represent hyperparameter configurations; the
    targets are the ``mean_loss`` column produced by the variance-budget
    generator.  No model fitting is needed — lookup is by nearest neighbor.
    """

    def __init__(
        self,
        generator: str,
        dataset: str,
        random_state: int = 42,
    ):
        self.generator = generator
        self.dataset = dataset
        self.storage_dir = Path(SyntheticGenerationParameters().storage_dir)
        self.random_state = random_state

        self.config_features = None
        self.mean_loss = None        # 1-D array of mean_loss values
        self.dataset_metadata = None
        self._initialized = False
        self._cat_col_categories: dict = {}  # col_name -> list[str]

    def initialize(self) -> None:
        """Load precomputed synthetic surrogate data."""
        if not self._initialized:
            storage = DatasetStorage(str(self.storage_dir))
            dataset_id = int(self.dataset)

            features_df, targets_df, metadata = storage.load_dataset(dataset_id)

            self.config_features = features_df
            self.dataset_metadata = metadata

            # Use mean_loss as the scalar performance target
            if "mean_loss" in targets_df.columns:
                self.mean_loss = targets_df["mean_loss"].to_numpy(dtype=np.float32)
            else:
                # Fallback: first target column (backwards compat with old schema)
                self.mean_loss = targets_df.iloc[:, 0].to_numpy(dtype=np.float32)

            # Record which columns are categorical (contain strings)
            for col in features_df.columns:
                if features_df[col].dtype == object:
                    self._cat_col_categories[col] = features_df[col].unique().tolist()

            self._initialized = True

    def predict(self, configuration: dict[str, Union[str, int, float, bool]]) -> float:
        """Predict performance by nearest-neighbor lookup in synthetic data."""
        self.initialize()

        config_vec = self._config_to_vector(configuration)
        config_matrix = self._feature_matrix()

        distances = np.linalg.norm(config_matrix - config_vec, axis=1)
        nearest_idx = np.argmin(distances)

        return float(self.mean_loss[nearest_idx])

    def _feature_matrix(self) -> np.ndarray:
        """Return numeric matrix of all stored configurations."""
        cols = []
        for col in self.config_features.columns:
            series = self.config_features[col]
            if series.dtype == object:
                cats = self._cat_col_categories.get(col, series.unique().tolist())
                cat_map = {c: float(i) for i, c in enumerate(cats)}
                cols.append(series.map(cat_map).fillna(0.0).to_numpy(dtype=np.float32))
            else:
                cols.append(series.to_numpy(dtype=np.float32))
        return np.stack(cols, axis=1)

    def _config_to_vector(self, configuration: dict) -> np.ndarray:
        """Convert configuration dict to numeric feature vector."""
        feature_names = self.config_features.columns.tolist()

        feature_vector = []
        for feature_name in feature_names:
            if feature_name in configuration:
                val = configuration[feature_name]
                if isinstance(val, (int, float, bool)):
                    feature_vector.append(float(val))
                else:
                    # Categorical: map to same integer code as in the stored data
                    cats = self._cat_col_categories.get(feature_name, [])
                    if val in cats:
                        feature_vector.append(float(cats.index(val)))
                    else:
                        try:
                            feature_vector.append(float(val))
                        except (ValueError, TypeError):
                            feature_vector.append(0.0)
            else:
                feature_vector.append(0.0)

        return np.array(feature_vector, dtype=np.float32).reshape(1, -1)
    
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
