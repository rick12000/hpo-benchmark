import numpy as np
from typing import Union, Dict, Any

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

        # Only initialize benchmark if not lazy loading:
        if not self._lazy:
            self._initialize_generator()
        else:
            self.generator = None

    def _initialize_generator(self) -> None:
        """Initialize the JAHS-201 benchmark generator if not already initialized.

        Returns:
            None
        """
        if not self._initialized:
            self.generator = Benchmark(
                task=self._dataset, lazy=False, metrics=self._metrics
            )
            self._initialized = True

    def initialize(self) -> None:
        """Public method to initialize the generator.

        Returns:
            None
        """
        self._initialize_generator()

    def predict(self, configuration: dict[str, Union[str, int, float, bool]]) -> float:
        """Return the negative validation accuracy for the given configuration.

        Args:
            configuration: Dictionary mapping parameter names to their values.

        Returns:
            Negative validation accuracy.
        """
        self._initialize_generator()
        return -self.generator(configuration)[200]["valid-acc"]

    def predict_runtime(
        self, configuration: dict[str, Union[str, int, float, bool]]
    ) -> float:
        """Return the runtime for the given configuration.

        Args:
            configuration: Dictionary mapping parameter names to their values.

        Returns:
            Runtime.
        """
        self._initialize_generator()
        return self.generator(configuration)[200]["runtime"]


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

    def is_parameter_active(self, configuration: dict, param_name: str) -> bool:
        """Check if a parameter is active in the configuration based on conditional dependencies.

        Args:
            configuration: Dictionary mapping parameter names to their values.
            param_name: Name of the parameter to check.

        Returns:
            True if the parameter is active, False otherwise.
        """
        conditions = self.config_space.get_conditions()
        param_conditions = [c for c in conditions if c.child.name == param_name]

        for condition in param_conditions:
            parent_name = condition.parent.name
            if parent_name not in configuration:
                return False
            parent_value = configuration[parent_name]
            if not condition.evaluate({parent_name: parent_value}):
                return False

        return True

    def _get_filtered_configuration(self, configuration: dict) -> dict:
        """Filter the configuration to include only active and fidelity parameters.

        Args:
            configuration: Dictionary mapping parameter names to their values.

        Returns:
            Filtered configuration dictionary including only active and fidelity parameters.
        """
        filtered_configuration = {}
        for param_name, param_value in configuration.items():
            if self.is_parameter_active(configuration, param_name):
                filtered_configuration[param_name] = param_value

        # Add fidelity parameters if they exist:
        if self.fidelity_space:
            for (
                fidelity_param_name,
                fidelity_param_value,
            ) in self.fidelity_space.items():
                filtered_configuration[fidelity_param_name] = fidelity_param_value

        filtered_configuration[self.instance_name] = self.generator.instance
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
        if "auc" in results:
            return -results["acc"]
        else:
            return -results["val_accuracy"]

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
