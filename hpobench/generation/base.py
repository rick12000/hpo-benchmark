from abc import ABC, abstractmethod
from typing import Union


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
