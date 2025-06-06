import numpy as np
from typing import Union, Optional, Dict, Any

from jahs_bench import Benchmark
from abc import ABC, abstractmethod

from yahpo_gym import local_config
from yahpo_gym import BenchmarkSet

local_config.init_config()
local_config.set_data_path("yahpo_bench_data")


def rastrigin(x, A=20):
    n = len(x)
    rastrigin_value = A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    return rastrigin_value


def ackley(x, a=20, b=0.2, c=2 * np.pi):
    n = len(x)
    term1 = -a * np.exp(-b * np.sqrt(np.sum(x**2) / n))
    term2 = -np.exp(np.sum(np.cos(c * x)) / n)
    ackley_value = term1 + term2 + a + np.exp(1)
    return ackley_value


def griewank(x):
    n = len(x)
    term1 = np.sum(x**2) / 4000
    term2 = 1
    for i in range(n):
        term2 *= np.cos(x[i] / np.sqrt(i + 1))
    griewank_value = term1 - term2 + 1
    return griewank_value


def weierstrass(x, a=0.5, b=3, kmax=20):
    n = len(x)
    weierstrass_value = 0
    for i in range(n):
        for k in range(kmax + 1):
            weierstrass_value += (a**k) * np.cos(2 * np.pi * (b**k) * (x[i] + 0.5))
        for k in range(kmax + 1):
            weierstrass_value -= (a**k) * np.cos(2 * np.pi * (b**k) * 0.5)
    return weierstrass_value


def shekel(x, m=10):  # m is the number of local minima
    n = len(x)
    A = np.random.rand(m, n) * 10  # random A matrix for each run
    C = np.random.rand(m) * 10
    shekel_value = 0
    for i in range(m):
        shekel_value -= 1 / (C[i] + np.sum((x - A[i]) ** 2))
    return -shekel_value


def hartmann6(x):
    alpha = [1.0, 1.2, 3.0, 3.2]
    A = np.array(
        [
            [1.0, 1.2, 3.0, 3.2],
            [3.6, 1.6, 0.7, 3.9],
            [4.0, 1.6, 0.8, 3.4],
            [1.6, 0.0, 3.6, 0.8],
            [1.6, 0.0, 3.6, 0.8],
        ]
    )
    P = np.array(
        [
            [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5894],
            [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
            [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
            [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
        ]
    )
    hartmann6_value = 0
    for i in range(4):
        inner_sum = 0
        for j in range(6):
            inner_sum += A[i, j] * (x[j] - P[i, j]) ** 2
        hartmann6_value -= alpha[i] * np.exp(-inner_sum)
    return -hartmann6_value


class ObjectiveMetricGenerator(ABC):
    def __init__(self, generator: str):
        self.generator = generator

    @abstractmethod
    def predict(self, configuration: dict[str, Union[str, int, float, bool]]):
        pass

    @abstractmethod
    def predict_runtime(self, configuration: dict[str, Union[str, int, float, bool]]):
        pass

    def initialize(self):
        """Initialize the generator if needed. Default implementation does nothing."""
        pass


class BlackBoxGenerator(ObjectiveMetricGenerator):
    def __init__(self, generator: str):
        self.generator = generator

    def predict(self, configuration: dict[str, Union[str, int, float, bool]]):
        # x = np.array(list(params.values()))
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

    def predict_runtime(self, configuration: dict[str, Union[str, int, float, bool]]):
        return 0


class Jahs201Generator(ObjectiveMetricGenerator):
    def __init__(
        self,
        dataset: str,
        metrics: list[str] = ["valid-acc", "runtime"],
        lazy: bool = True,  # Default to lazy loading
    ):
        # Store dataset and metrics
        self._dataset = dataset
        self._metrics = metrics
        self._lazy = lazy
        self._initialized = False
        # Only initialize benchmark if not lazy
        if not self._lazy:
            self._initialize_generator()
        else:
            self.generator = None

    def _initialize_generator(self):
        """Initialize the JAHS-201 benchmark generator if not already initialized."""
        if not self._initialized:
            self.generator = Benchmark(
                task=self._dataset, lazy=False, metrics=self._metrics
            )
            self._initialized = True

    def initialize(self):
        """Public method to initialize the generator."""
        self._initialize_generator()

    def predict(self, configuration: dict[str, Union[str, int, float, bool]]):
        self._initialize_generator()
        return -self.generator(configuration)[200]["valid-acc"]

    def predict_runtime(self, configuration: dict[str, Union[str, int, float, bool]]):
        self._initialize_generator()
        return self.generator(configuration)[200]["runtime"]


class YahpoGenerator(ObjectiveMetricGenerator):
    """Class for wrapping yahpo gym generators."""

    def __init__(
        self,
        dataset: str,
        instance_value: Any,
        instance_name: str,
        fidelity_space: Dict,
        config_space=None,
    ):
        """Initialize YAHPO generator.

        Args:
            dataset: YAHPO dataset name
            instance_value: YAHPO instance value
            instance_name: Name of the instance parameter
            fidelity_space: Dictionary of fidelity parameters and their values
            config_space: Full ConfigSpace object from YAHPO with conditions
        """
        self.dataset = dataset
        self.instance_name = instance_name
        self.generator = BenchmarkSet(dataset, instance=instance_value)

        # Store the full ConfigSpace for validation
        self.config_space = config_space

        # Store fidelity parameters
        self.fidelity_space = fidelity_space

    def is_parameter_active(self, configuration: dict, param_name: str) -> bool:
        """Check if a parameter should be active given the configuration.

        Args:
            configuration: Configuration dictionary
            param_name: Parameter name to check

        Returns:
            True if the parameter should be active, False otherwise
        """
        if self.config_space is None:
            return True  # If no config space, assume all parameters are active

        # Get conditions for this parameter
        conditions = self.config_space.get_conditions()
        param_conditions = [c for c in conditions if c.child.name == param_name]

        if not param_conditions:
            return True  # If no conditions, parameter is always active

        # Check all conditions for this parameter
        for condition in param_conditions:
            parent_name = condition.parent.name
            if parent_name not in configuration:
                return False

            parent_value = configuration[parent_name]

            # Check if condition is satisfied
            if not condition.evaluate({parent_name: parent_value}):
                return False

        return True

    def predict(self, configuration):
        """Predict the performance of a configuration using the YAHPO generator.

        Args:
            configuration: Configuration to evaluate

        Returns:
            Negative validation accuracy (to be minimized)
        """
        # Create a filtered configuration with only active parameters
        filtered_configuration = {}

        # Copy active parameters from the input configuration
        for param_name, param_value in configuration.items():
            if self.is_parameter_active(configuration, param_name):
                filtered_configuration[param_name] = param_value

        # Add fidelity parameters
        if self.fidelity_space is not None:
            for (
                fidelity_param_name,
                fidelity_param_value,
            ) in self.fidelity_space.items():
                filtered_configuration[fidelity_param_name] = fidelity_param_value

        # Add instance name parameter
        if self.instance_name:
            filtered_configuration[self.instance_name] = self.generator.instance

        # Call the objective function
        objective_results = self.generator.objective_function(filtered_configuration)[0]
        if "auc" in objective_results:
            return -objective_results["acc"]
        else:
            return -objective_results["val_accuracy"]

    def predict_runtime(self, configuration: dict[str, Union[str, int, float, bool]]):
        # Filter the configuration to only include active parameters
        filtered_configuration = {}
        for param_name, param_value in configuration.items():
            if self.is_parameter_active(configuration, param_name):
                filtered_configuration[param_name] = param_value

        # Add fidelity parameters
        if self.fidelity_space is not None:
            for (
                fidelity_param_name,
                fidelity_param_value,
            ) in self.fidelity_space.items():
                filtered_configuration[fidelity_param_name] = fidelity_param_value

        # Add instance name
        filtered_configuration[self.instance_name] = self.generator.instance

        results = self.generator.objective_function(filtered_configuration)[0]
        if "time" in results:
            return results["time"]
        elif "runtime" in results:
            return results["runtime"]
        else:
            return results["timetrain"] + results["timepredict"]
