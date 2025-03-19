from sympy import N
from hpobench.prepare import (
    setup_yahpo_instance_configs,
    setup_jahs201_configs,
    setup_blackbox_configs,
)
from hpobench.config import (
    ExperimentConfig,
    JAHS201_SEARCH_SPACE,
    BLACK_BOX_SEARCH_SPACE,
)
from hpobench.config import DEV_TUNING_CONFIGURATIONS


def test_setup_yahpo_instance_configs():

    n_instances = 5

    configs = setup_yahpo_instance_configs(
        dataset="lcbench",
        tuning_configurations=DEV_TUNING_CONFIGURATIONS,
        n_warm_starts=5,
        n_trials=10,
        timeout=3600,
        max_n_instances=n_instances,
    )

    assert len(configs) == n_instances
    assert all(isinstance(c, ExperimentConfig) for c in configs)
    assert all(c.benchmark_identifier == "lcbench" for c in configs)


def test_setup_jahs201_configs():
    datasets = ["cifar10"]

    configs = setup_jahs201_configs(
        datasets=datasets,
        tuning_configurations=DEV_TUNING_CONFIGURATIONS,
        n_warm_starts=5,
        n_trials=10,
        timeout=3600,
    )

    assert len(configs) == len(datasets)
    assert all(isinstance(c, ExperimentConfig) for c in configs)
    assert all(c.search_space == JAHS201_SEARCH_SPACE for c in configs)
    assert all(c.benchmark_identifier == "JAHS-201" for c in configs)
    assert [c.dataset_identifier for c in configs] == datasets


def test_setup_blackbox_configs():
    functions = ["hartmann"]

    configs = setup_blackbox_configs(
        functions=functions,
        tuning_configurations=DEV_TUNING_CONFIGURATIONS,
        n_warm_starts=5,
        n_trials=10,
        timeout=3600,
    )

    assert len(configs) == len(functions)
    assert all(isinstance(c, ExperimentConfig) for c in configs)
    assert all(c.search_space == BLACK_BOX_SEARCH_SPACE for c in configs)
    assert all(c.benchmark_identifier == "blackbox" for c in configs)
    assert [c.dataset_identifier for c in configs] == functions
