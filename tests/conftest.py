import numpy as np
import pytest
import pandas as pd
from hpobench.generation.generate import BlackBoxGenerator
from hpobench.config.config_types import FloatRange


@pytest.fixture
def toy_relativized_runtime_data():
    data = {
        "benchmark_identifier": [
            "bench_A",
            "bench_A",
            "bench_A",
            "bench_A",
            "bench_A",
            "bench_A",
            "bench_A",
            "bench_A",
            "bench_A",
            "bench_B",
            "bench_B",
            "bench_B",
            "bench_B",
            "bench_B",
            "bench_B",
            "bench_B",
            "bench_B",
            "bench_B",
        ],
        "dataset": [
            "dataset_1",
            "dataset_1",
            "dataset_1",
            "dataset_2",
            "dataset_2",
            "dataset_2",
            "dataset_3",
            "dataset_3",
            "dataset_3",
            "dataset_4",
            "dataset_4",
            "dataset_4",
            "dataset_5",
            "dataset_5",
            "dataset_5",
            "dataset_6",
            "dataset_6",
            "dataset_6",
        ],
        "tuner": [
            "tuner_X",
            "tuner_Y",
            "tuner_Z",
            "tuner_X",
            "tuner_Y",
            "tuner_Z",
            "tuner_X",
            "tuner_Y",
            "tuner_Z",
            "tuner_X",
            "tuner_Y",
            "tuner_Z",
            "tuner_X",
            "tuner_Y",
            "tuner_Z",
            "tuner_X",
            "tuner_Y",
            "tuner_Z",
        ],
        "sampler": [
            "TPE",
            "TPE",
            "Random",
            "Random",
            "TPE",
            "Random",
            "TPE",
            "Random",
            "TPE",
            "TPE",
            "Random",
            "TPE",
            "Random",
            "TPE",
            "Random",
            "TPE",
            "Random",
            "TPE",
        ],
        "confidence_level": [
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
        ],
        "rank": [1, 2, 3, 1.5, 1.5, 3, 2, 3, 1, 1, 2, 3, 3, 1.5, 1.5, 3, 1, 2],
    }

    return pd.DataFrame(data)


@pytest.fixture
def grouping_columns():
    return [
        "benchmark_identifier",
        "dataset",
        "tuner",
        "repetition",
        "sampler",
        "confidence_level",
        "estimator_architecture",
    ]


@pytest.fixture
def repetition_column():
    return "repetition"


@pytest.fixture
def performance_column():
    return "performance"


@pytest.fixture
def tuner_column():
    return "tuner"


@pytest.fixture
def sampler_column():
    return "sampler"


@pytest.fixture
def confidence_level_column():
    return "confidence_level"


@pytest.fixture
def estimator_architecture_column():
    return "estimator_architecture"


@pytest.fixture
def benchmark_column():
    return "benchmark_identifier"


@pytest.fixture
def budget_units():
    return ["iteration", "runtime"]


@pytest.fixture
def dataset_aggregators():
    return ["benchmark_identifier", "dataset"]


@pytest.fixture
def entity_columns():
    return ["benchmark_identifier", "dataset", "tuner"]


@pytest.fixture
def dummy_blackbox_generator():
    return {
        "generator": "rastrigin",
    }


@pytest.fixture
def dummy_yahpo_generator():
    return {
        "dataset": "167168",
    }


@pytest.fixture
def small_param_space():
    """Create a small parameter search space for testing."""
    return {
        "x": FloatRange(lower=0, upper=100.0),
        "y": FloatRange(lower=0, upper=100.0),
    }


@pytest.fixture
def performance_generator():
    """Create a BlackBoxGenerator with rastrigin for testing."""
    return BlackBoxGenerator(generator="rastrigin")


@pytest.fixture
def warm_start_configs(performance_generator):
    """Create a set of warm start configurations for testing with actual performance values."""
    configs = [
        {"x": 0.0, "y": 0.0},
        {"x": 1.0, "y": 1.0},
        {"x": 10.0, "y": 20.0},
        {"x": 30.0, "y": 40.0},
        {"x": 50.0, "y": 60.0},
        {"x": 70.0, "y": 80.0},
        {"x": 90.0, "y": 100.0},
        {"x": 75.0, "y": 25.0},
        {"x": 25.0, "y": 75.0},
        {"x": 45.0, "y": 55.0},
    ]

    # Get actual performances from the generator instead of hardcoding
    return [(config, performance_generator.predict(config)) for config in configs]


# Fixtures for metrics testing


@pytest.fixture
def extreme_significant_data():
    """Data where entities have extremely significant differences (zero variance)."""
    data = []
    # Use more datasets to get extremely small p-values in Wilcoxon test
    datasets = [f"dataset{i}" for i in range(1, 21)]  # 20 datasets

    # Entity A always ranks 1, B always ranks 2, C always ranks 3
    for dataset in datasets:
        data.extend(
            [
                {"dataset": dataset, "entity": "entity_A", "rank": 1.0},
                {"dataset": dataset, "entity": "entity_B", "rank": 2.0},
                {"dataset": dataset, "entity": "entity_C", "rank": 3.0},
            ]
        )

    return pd.DataFrame(data)


@pytest.fixture
def identical_ranks_data():
    """Data where all entities have identical ranks (no differences)."""
    data = []
    datasets = [f"dataset{i}" for i in range(1, 21)]  # 20 datasets
    entities = ["entity_A", "entity_B", "entity_C"]

    # All entities always get the same rank
    for dataset in datasets:
        for entity in entities:
            data.append({"dataset": dataset, "entity": entity, "rank": 2.0})

    return pd.DataFrame(data)


@pytest.fixture
def realistic_significant_data():
    """Realistic data with significant but variable differences."""
    np.random.seed(42)
    data = []
    datasets = [f"dataset{i}" for i in range(1, 21)]  # 20 datasets

    for dataset in datasets:
        # Entity A is consistently better (lower ranks) but with some variance
        rank_A = np.random.normal(1.2, 0.2)
        # Entity B is consistently middle
        rank_B = np.random.normal(2.5, 0.3)
        # Entity C is consistently worse
        rank_C = np.random.normal(3.8, 0.2)

        data.extend(
            [
                {"dataset": dataset, "entity": "entity_A", "rank": max(1.0, rank_A)},
                {"dataset": dataset, "entity": "entity_B", "rank": rank_B},
                {"dataset": dataset, "entity": "entity_C", "rank": max(1.0, rank_C)},
            ]
        )

    return pd.DataFrame(data)


@pytest.fixture
def realistic_insignificant_data():
    """Realistic data with small, insignificant differences."""
    np.random.seed(123)
    data = []
    datasets = [f"dataset{i}" for i in range(1, 21)]  # 20 datasets

    for dataset in datasets:
        # All entities have similar performance with high variance
        base_rank = np.random.normal(2.0, 0.1)
        data.extend(
            [
                {
                    "dataset": dataset,
                    "entity": "entity_A",
                    "rank": max(1.0, base_rank + np.random.normal(0, 0.4)),
                },
                {
                    "dataset": dataset,
                    "entity": "entity_B",
                    "rank": max(1.0, base_rank + np.random.normal(0, 0.4)),
                },
                {
                    "dataset": dataset,
                    "entity": "entity_C",
                    "rank": max(1.0, base_rank + np.random.normal(0, 0.4)),
                },
            ]
        )

    return pd.DataFrame(data)


@pytest.fixture
def grouped_test_data():
    """Data with breakout groups for testing grouped statistical tests."""
    data = []
    datasets = [f"dataset{i}" for i in range(1, 21)]  # 20 datasets
    entities = ["entity_A", "entity_B", "entity_C"]
    groups = ["group1", "group2"]

    for group in groups:
        for dataset in datasets:
            if group == "group1":
                # Group 1: significant differences
                ranks = {"entity_A": 1.0, "entity_B": 2.0, "entity_C": 3.0}
            else:
                # Group 2: no differences
                ranks = {"entity_A": 2.0, "entity_B": 2.0, "entity_C": 2.0}

            for entity in entities:
                data.append(
                    {
                        "dataset": dataset,
                        "entity": entity,
                        "rank": ranks[entity],
                        "group": group,
                    }
                )

    return pd.DataFrame(data)


@pytest.fixture
def insufficient_data():
    """Data with insufficient samples for statistical tests."""
    return pd.DataFrame(
        [
            {"dataset": "dataset1", "entity": "entity_A", "rank": 1.0},
            {"dataset": "dataset1", "entity": "entity_B", "rank": 2.0},
        ]
    )


@pytest.fixture
def independent_X_y_data():
    """X features completely independent of y outcome for likelihood ratio testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 3

    # Independent random features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=["feature1", "feature2", "feature3"],
    )
    # Random binary outcome
    y = pd.Series(np.random.binomial(1, 0.5, n_samples))

    return X, y


@pytest.fixture
def dependent_X_y_data():
    """X features with functional relationship to y outcome for likelihood ratio testing."""
    np.random.seed(42)
    n_samples = 100

    # Create features with relationship to outcome
    feature1 = np.random.randn(n_samples)
    feature2 = np.random.randn(n_samples)
    feature3 = np.random.randn(n_samples)

    # Create y with strong relationship to features
    linear_combination = 2 * feature1 + 1.5 * feature2 - 0.8 * feature3
    probabilities = 1 / (1 + np.exp(-linear_combination))  # sigmoid
    y = pd.Series(np.random.binomial(1, probabilities))

    X = pd.DataFrame({"feature1": feature1, "feature2": feature2, "feature3": feature3})

    return X, y


@pytest.fixture
def single_class_y_data():
    """Data with only one class in y for testing edge cases."""
    np.random.seed(42)
    n_samples = 50
    n_features = 3

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=["feature1", "feature2", "feature3"],
    )
    # All outcomes are the same class
    y = pd.Series(np.ones(n_samples, dtype=int))

    return X, y
