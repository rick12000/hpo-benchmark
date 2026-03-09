import pytest
from hpobench.orchestration.prepare import (
    setup_yahpo_instance_configs,
    setup_blackbox_configs,
)
from hpobench.config.types import ExperimentConfig, CustomGPModel, TunerConfig
from hpobench.config.generator_metadata import BLACK_BOX_SEARCH_SPACE

DEV_TUNING_CONFIGURATIONS = [
    TunerConfig(
        tuner=CustomGPModel(backend="gp_opt", searcher="EI"),
        tuner_identifier="GP-EI",
    )
]


def test_setup_yahpo_instance_configs():
    n_instances = 5

    configs = setup_yahpo_instance_configs(
        benchmark="lcbench",
        tuning_configurations=DEV_TUNING_CONFIGURATIONS,
        max_n_instances=n_instances,
    )

    assert len(configs) == n_instances
    assert all(isinstance(c, ExperimentConfig) for c in configs)
    assert all(c.benchmark_identifier == "lcbench" for c in configs)


def test_setup_blackbox_configs():
    functions = ["hartmann"]

    configs = setup_blackbox_configs(
        functions=functions,
        tuning_configurations=DEV_TUNING_CONFIGURATIONS,
    )

    assert len(configs) == len(functions)
    assert all(isinstance(c, ExperimentConfig) for c in configs)
    assert all(c.search_space == BLACK_BOX_SEARCH_SPACE for c in configs)
    assert all(c.benchmark_identifier == "blackbox" for c in configs)
    assert [c.dataset_identifier for c in configs] == functions


@pytest.mark.parametrize(
    "benchmark,expected_max_fidelity",
    [
        ("lcbench", {"epoch": 50}),
        ("rbv2_aknn", {"repl": 10, "trainsize": 1.0}),
    ],
)
def test_yahpo_instance_configs_use_maximum_fidelity(benchmark, expected_max_fidelity):
    configs = setup_yahpo_instance_configs(
        benchmark=benchmark,
        tuning_configurations=DEV_TUNING_CONFIGURATIONS,
        max_n_instances=1,
    )

    assert len(configs) >= 1
    generator = configs[0].objective_function

    for param_name, expected_max in expected_max_fidelity.items():
        assert param_name in generator.fidelity_space
        actual_value = generator.fidelity_space[param_name]
        assert actual_value == expected_max
