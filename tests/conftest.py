import numpy as np
import pytest
import pandas as pd
from hpobench.generation.generate import BlackBoxGenerator
from hpobench.config.config_types import FloatRange


@pytest.fixture
def dummy_calibration_raw_data():
    # Build hierarchical realistic data:
    # - 1 benchmark identifier with 2 datasets
    # - 2 tuners per dataset, 2 repetitions per tuner
    # - 60 iterations per repetition (0..59)
    # - 2 confidence levels per iteration: 0.5 and 0.75 (creates two rows per iteration)
    benchmarks = ["bench_A"]
    datasets_per_benchmark = {
        "bench_A": ["dataset_a1", "dataset_a2"],
    }
    tuners = ["Conformalized + DtACI", "Conformalized"]
    sampler_map = {"Conformalized + DtACI": "ei", "Conformalized": "ei"}
    n_reps = 2
    n_iters = 91  # iterations 0..59
    confidences = [0.5, 0.75]

    data = {
        k: []
        for k in [
            "performance",
            "iteration",
            "runtime",
            "benchmark_identifier",
            "dataset",
            "tuner",
            "repetition",
            "sampler",
            "confidence_level",
            "estimator_architecture",
            "breach_status",
            "miscoverage_penalty",
            "winkler_score",
            "width",
            "tabularized_configuration",
        ]
    }

    # small set of tabular configurations to cycle through
    tabularized_configuration = [
        [343.0, 0.0143, 0.074, 290.0, 0.63, 2.0, 0.019],
        [439.0, 0.0184, 0.303, 142.0, 0.40, 4.0, 0.073],
        [59.0, 0.0813, 0.569, 146.0, 0.15, 2.0, 0.087],
        [337.0, 0.0436, 0.712, 433.0, 0.93, 3.0, 0.087],
        [347.0, 0.0016, 0.084, 276.0, 0.50, 1.0, 0.007],
    ]

    row_idx = 0
    for b in benchmarks:
        for d in datasets_per_benchmark[b]:
            # choose a base runtime per dataset so that runtimes increase with iteration
            base_runtime = 50 if b == "bench_B" else 200
            per_iter_increment = 5.0
            for t in tuners:
                sampler = sampler_map.get(t, "random")
                for rep in range(1, n_reps + 1):
                    # estimator architecture related to tuner
                    est_arch = f"{t}_arch_v{rep}"
                    for it in range(0, n_iters):
                        # create one row per confidence level
                        for conf in confidences:
                            # performance in [0.7, 0.9]
                            perf = float(np.round(np.random.uniform(0.7, 0.9), 4))

                            # runtime strictly increasing with iteration (small positive jitter keeps monotonicity)
                            jitter = np.random.uniform(0.0, 0.01)
                            runtime = float(
                                base_runtime
                                + it * per_iter_increment
                                + rep * 0.1
                                + jitter
                            )

                            # breach_status and related metrics empty when iteration == 0
                            if it == 0:
                                breach = np.nan
                                miscoverage = np.nan
                                winkler = np.nan
                                width = np.nan
                            else:
                                # assign breach randomly with 50/50 chance across rows for non-zero iterations
                                # use a Bernoulli(0.5) draw so values are 0 or 1
                                breach = int(np.random.binomial(1, 0.5))
                                # vary numeric scores across rows; small floats
                                tuner_offset = (
                                    0.01 if t == "Conformalized + DtACI" else 0.02
                                )
                                miscoverage = float(
                                    np.round(
                                        0.05
                                        + rep * 0.01
                                        + it * 0.001
                                        + tuner_offset
                                        + np.random.uniform(-0.005, 0.005),
                                        6,
                                    )
                                )
                                winkler = float(
                                    np.round(
                                        0.1
                                        + rep * 0.02
                                        + it * 0.002
                                        + tuner_offset
                                        + np.random.uniform(-0.01, 0.01),
                                        6,
                                    )
                                )
                                width = float(
                                    np.round(
                                        0.2
                                        + rep * 0.03
                                        + it * 0.003
                                        + tuner_offset
                                        + np.random.uniform(-0.01, 0.01),
                                        6,
                                    )
                                )

                            data["performance"].append(perf)
                            data["iteration"].append(it)
                            data["runtime"].append(runtime)
                            data["benchmark_identifier"].append(b)
                            data["dataset"].append(d)
                            data["tuner"].append(t)
                            data["repetition"].append(rep)
                            data["sampler"].append(sampler)
                            data["confidence_level"].append(conf)
                            data["estimator_architecture"].append(est_arch)
                            data["breach_status"].append(breach)
                            data["miscoverage_penalty"].append(miscoverage)
                            data["winkler_score"].append(winkler)
                            data["width"].append(width)
                            data["tabularized_configuration"].append(
                                tabularized_configuration[
                                    row_idx % len(tabularized_configuration)
                                ]
                            )

                            row_idx += 1

    df_data = pd.DataFrame(data).sort_values(
        by=[
            "benchmark_identifier",
            "dataset",
            "confidence_level",
            "tuner",
            "repetition",
            "iteration",
        ]
    )
    return df_data


def dummy_processing_raw_data():
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
        "sampler": ["gbrt"] * 10 + ["tpe"] * 9 + ["gbrt"] * 4 + ["tpe"] * 4,
        "confidence_level": [None]
        * 27,  # Non-confopt tuners have None confidence level
        "estimator_architecture": [None]
        * 27,  # Non-confopt tuners have None estimator architecture
        # Added columns to match dummy_calibration_raw_data fixture
        "breach_status": [
            "",  # iteration 1 -> empty
            1,
            0,
            1,
            0,
            1,
            "",  # iteration 1
            0,
            1,
            0,
            "",  # iteration 1
            1,
            0,
            1,
            0,
            "",  # iteration 1
            1,
            0,
            1,
            "",  # iteration 1
            0,
            "",  # iteration 1
            1,
            "",  # iteration 1
            0,
            "",  # iteration 1
            1,
        ],
        "miscoverage_penalty": [
            np.nan,  # matches breach == ""
            0.01,
            0.02,
            0.03,
            0.04,
            0.05,
            np.nan,
            0.01,
            0.02,
            0.03,
            np.nan,
            0.01,
            0.02,
            0.03,
            0.04,
            np.nan,
            0.01,
            0.02,
            0.03,
            np.nan,
            0.01,
            np.nan,
            0.02,
            np.nan,
            0.01,
            np.nan,
            0.02,
        ],
        "winkler_score": [
            np.nan,
            0.10,
            0.12,
            0.15,
            0.18,
            0.2,
            np.nan,
            0.11,
            0.13,
            0.16,
            np.nan,
            0.10,
            0.12,
            0.15,
            0.18,
            np.nan,
            0.11,
            0.13,
            0.16,
            np.nan,
            0.10,
            np.nan,
            0.12,
            np.nan,
            0.11,
            np.nan,
            0.13,
        ],
        "width": [
            np.nan,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            np.nan,
            0.22,
            0.27,
            0.32,
            np.nan,
            0.21,
            0.26,
            0.31,
            0.36,
            np.nan,
            0.22,
            0.27,
            0.33,
            np.nan,
            0.20,
            np.nan,
            0.25,
            np.nan,
            0.22,
            np.nan,
            0.27,
        ],
        "tabularized_configuration": [
            [343.0, 0.0143, 0.074, 290.0, 0.63, 2.0, 0.019],
            [439.0, 0.0184, 0.303, 142.0, 0.40, 4.0, 0.073],
            [59.0, 0.0813, 0.569, 146.0, 0.15, 2.0, 0.087],
            [337.0, 0.0436, 0.712, 433.0, 0.93, 3.0, 0.087],
            [347.0, 0.0016, 0.084, 276.0, 0.50, 1.0, 0.007],
            [343.0, 0.0143, 0.074, 290.0, 0.63, 2.0, 0.019],
            [439.0, 0.0184, 0.303, 142.0, 0.40, 4.0, 0.073],
            [59.0, 0.0813, 0.569, 146.0, 0.15, 2.0, 0.087],
            [337.0, 0.0436, 0.712, 433.0, 0.93, 3.0, 0.087],
            [347.0, 0.0016, 0.084, 276.0, 0.50, 1.0, 0.007],
            [343.0, 0.0143, 0.074, 290.0, 0.63, 2.0, 0.019],
            [439.0, 0.0184, 0.303, 142.0, 0.40, 4.0, 0.073],
            [59.0, 0.0813, 0.569, 146.0, 0.15, 2.0, 0.087],
            [337.0, 0.0436, 0.712, 433.0, 0.93, 3.0, 0.087],
            [347.0, 0.0016, 0.084, 276.0, 0.50, 1.0, 0.007],
            [343.0, 0.0143, 0.074, 290.0, 0.63, 2.0, 0.019],
            [439.0, 0.0184, 0.303, 142.0, 0.40, 4.0, 0.073],
            [59.0, 0.0813, 0.569, 146.0, 0.15, 2.0, 0.087],
            [337.0, 0.0436, 0.712, 433.0, 0.93, 3.0, 0.087],
            [347.0, 0.0016, 0.084, 276.0, 0.50, 1.0, 0.007],
            [343.0, 0.0143, 0.074, 290.0, 0.63, 2.0, 0.019],
            [439.0, 0.0184, 0.303, 142.0, 0.40, 4.0, 0.073],
            [59.0, 0.0813, 0.569, 146.0, 0.15, 2.0, 0.087],
            [337.0, 0.0436, 0.712, 433.0, 0.93, 3.0, 0.087],
            [347.0, 0.0016, 0.084, 276.0, 0.50, 1.0, 0.007],
            [343.0, 0.0143, 0.074, 290.0, 0.63, 2.0, 0.019],
            [439.0, 0.0184, 0.303, 142.0, 0.40, 4.0, 0.073],
        ],
    }
    return pd.DataFrame(data)


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
def dummy_jahs201_generator():
    return {
        "dataset": "cifar10",
    }


@pytest.fixture
def dummy_yahpo_generator():
    return {
        "dataset": "167168",
    }


@pytest.fixture
def dummy_nas301_generator():
    return {
        "instance_value": "CIFAR10",
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
