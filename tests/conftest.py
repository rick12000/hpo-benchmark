import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def dummy_experiment_data():
    data = {
        "performance": [
            -83.3992,
            -91.3902,
            -89.9583,
            -88.5413,
            -79.0829,
            -87.9277,
            -80.241,
            -87.4604,
            -84.0319,
            -71.6547,
            -89.9583,
            -88.5413,
            -79.0829,
            -83.757,
            -81.358,
            -87.4604,
            -84.0319,
            -71.6547,
            -84.2562,
            -84.0319,
            -71.6547,
            -89.9583,
            -88.5413,
            -79.0829,
            -88.5413,
            -79.0829,
            -83.757,
        ],
        "iteration": [
            1,
            2,
            3,
            4,
            5,
            6,
            1,
            2,
            3,
            4,
            1,
            2,
            3,
            4,
            5,
            1,
            2,
            3,
            4,
            1,
            2,
            1,
            2,
            1,
            2,
            1,
            2,
        ],
        "breach_status": [
            np.nan,
            1,
            0,
            0,
            1,
            1,
            0,
            0,
            1,
            1,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ],
        "runtime": [
            256.8546,
            257.8,
            259.8,
            259.9,
            260,
            260.1,
            254.2,
            257.5,
            258,
            259,
            257.1,
            258.2,
            258.3,
            258.5,
            259.1,
            256,
            257.2,
            258.5,
            258.9,
            50,
            50.5,
            50.1,
            52,
            50.3,
            52.2,
            50.3,
            50.8,
        ],
        "benchmark_identifier": ["lcbench"] * 19 + ["nahs201"] * 8,
        "dataset": [3945] * 19 + ["cifar10"] * 8,
        "tuner": ["GBRT"] * 10 + ["TPE"] * 9 + ["GBRT"] * 4 + ["TPE"] * 4,
        "repetition": [1] * 6
        + [2] * 4
        + [1] * 5
        + [2] * 4
        + [1] * 2
        + [2] * 2
        + [1] * 2
        + [2] * 2,
    }
    df_data = pd.DataFrame(data).sample(frac=1)

    return df_data


@pytest.fixture
def grouping_columns():
    return ["benchmark_identifier", "dataset", "tuner", "repetition"]


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
def dummy_jahs201_generator():
    return {
        "dataset": "cifar10",
    }


@pytest.fixture
def dummy_yahpo_generator():
    return {
        "dataset": "167168",
    }
