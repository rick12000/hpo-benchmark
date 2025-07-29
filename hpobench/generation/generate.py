import numpy as np
from typing import Union, Dict, Any

from ConfigSpace import Configuration

from jahs_bench import Benchmark
from abc import ABC, abstractmethod

from yahpo_gym import local_config
from yahpo_gym import BenchmarkSet

from hpobench.generation.black_box_functions import (
    rastrigin,
    ackley,
    griewank,
    weierstrass,
    shekel,
    hartmann6,
)

local_config.init_config()
local_config.set_data_path("yahpo_bench_data")


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
    def predict_runtime(
        self, configuration: dict[str, Union[str, int, float, bool]]
    ) -> float:
        """Return the runtime for a given configuration.

        Args:
            configuration: Dictionary mapping parameter names to their values.

        Returns:
            The runtime for the given configuration.
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

    def predict(self, configuration: dict[str, Union[str, int, float, bool]]) -> float:
        """Evaluate the black-box function for the given configuration.

        Args:
            configuration: Dictionary mapping parameter names to their values.

        Returns:
            The function value for the given configuration.
        """
        x = np.array(list(configuration.values()), dtype=float)

        if self.generator == "rastrigin":
            y = rastrigin(x=x)
        elif self.generator == "ackley":
            y = ackley(x=x)
        elif self.generator == "griewank":
            y = griewank(x=x)
        elif self.generator == "weierstrass":
            y = weierstrass(x=x)
        elif self.generator == "shekel":
            y = shekel(x=x)
        elif self.generator == "hartmann6":
            y = hartmann6(x=x)
        else:
            raise ValueError(f"Unknown generator: {self.generator}")
        return y

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


class Jahs201Generator(ObjectiveMetricGenerator):
    """Objective metric generator for the JAHS-201 surrogate benchmark.

    Args:
        dataset: Name of the JAHS-201 dataset.
        metrics: List of metric names to use.
        lazy: If True, defer initialization of the generator until first use.
    """

    def __init__(
        self,
        dataset: str,
        metrics: list[str] = ["valid-acc", "runtime"],
        lazy: bool = True,
    ):
        self._dataset = dataset
        self._metrics = metrics
        self._lazy = lazy
        self._initialized = False

        self.default_fidelities = {
            "epoch": 200,
            "W": 16,
            "N": 5,
            "Resolution": 1,
        }

        if not self._lazy:
            self._initialize_generator()
        else:
            self.generator = None

    def _initialize_generator(self) -> None:
        if not self._initialized:
            self.generator = Benchmark(
                task=self._dataset, lazy=False, metrics=self._metrics
            )
            self._initialized = True

    def initialize(self) -> None:
        self._initialize_generator()

    def _merge_with_fidelities(
        self, configuration: dict[str, Union[str, int, float, bool]]
    ) -> dict[str, Union[str, int, float, bool]]:
        merged = configuration.copy()
        merged.update(self.default_fidelities)
        return merged

    def predict(self, configuration: dict[str, Union[str, int, float, bool]]) -> float:
        self._initialize_generator()
        merged_config = self._merge_with_fidelities(configuration)
        return -self.generator(merged_config)[self.default_fidelities["epoch"]][
            "valid-acc"
        ]

    def predict_runtime(
        self, configuration: dict[str, Union[str, int, float, bool]]
    ) -> float:
        self._initialize_generator()
        merged_config = self._merge_with_fidelities(configuration)
        return self.generator(merged_config)[self.default_fidelities["epoch"]][
            "runtime"
        ]


class YahpoGenerator(ObjectiveMetricGenerator):
    """Objective metric generator for YAHPO Gym surrogate benchmarks.

    Handles instance-specific and fidelity-aware configuration evaluation.

    Args:
        dataset: Name of the YAHPO benchmark scenario.
        instance_value: Value of the instance for this experiment.
        instance_name: Name of the instance parameter in the configuration space.
        fidelity_space: Dictionary of fidelity parameter names and their values.
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
        self.dataset = dataset
        self.instance_name = instance_name
        self.generator = BenchmarkSet(dataset, instance=instance_value)

        self.config_space = config_space
        self.fidelity_space = fidelity_space

    def _get_filtered_configuration(self, configuration: dict) -> dict:
        """Filter the configuration to include only active and fidelity parameters.

        Uses ConfigSpace's built-in get_active_hyperparameters method for robust
        conditional dependency handling, aligned with Syne Tune's approach.

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
        filtered_configuration = self._get_filtered_configuration(configuration)

        # Call the objective function
        results = self.generator.objective_function(filtered_configuration)[0]
        if "val_accuracy" in results:
            return -results["val_accuracy"]
        elif "acc" in results:
            return -results["acc"]
        elif "auc" in results:
            return -results["auc"]
        else:
            raise ValueError(
                f"No suitable metric found in results: {list(results.keys())}"
            )

    def predict_runtime(
        self, configuration: dict[str, Union[str, int, float, bool]]
    ) -> float:
        """Return the runtime for the given configuration.

        Args:
            configuration: Dictionary mapping parameter names to their values.

        Returns:
            Runtime.
        """
        filtered_configuration = self._get_filtered_configuration(configuration)

        results = self.generator.objective_function(filtered_configuration)[0]
        if "time" in results:
            return results["time"]
        elif "runtime" in results:
            return results["runtime"]
        else:
            return results["timetrain"] + results["timepredict"]
