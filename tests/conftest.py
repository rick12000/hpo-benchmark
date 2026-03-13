import numpy as np
import pytest
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from hpobench.generation.generate import BlackBoxGenerator
from hpobench.config.types import FloatRange, IntRange, CategoricalRange, ExperimentConfig, TunerConfig, CustomGPModel, LTRConfig, LTRTuningConfig
from hpobench.config.schema import SurrogateMetafeaturesSchema, Aliases, BenchmarkDataSchema
from hpobench.utils import generate_hyperparameter_combinations


@pytest.fixture
def small_param_space():
    return {
        "x": FloatRange(lower=0, upper=100.0),
        "y": FloatRange(lower=0, upper=100.0),
    }


@pytest.fixture
def performance_generator():
    return BlackBoxGenerator(generator="rastrigin")


@pytest.fixture
def warm_start_configs(performance_generator):
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
    return [(config, performance_generator.predict(config)) for config in configs]


@pytest.fixture
def simple_search_space():
    return {
        "lr": FloatRange(lower=1e-5, upper=1e-1),
        "n_layers": IntRange(lower=1, upper=5),
        "activation": CategoricalRange(choices=["relu", "tanh"]),
        "optimizer": CategoricalRange(choices=["adam", "sgd", "rmsprop"]),
    }


@pytest.fixture
def simple_configs(simple_search_space):
    return generate_hyperparameter_combinations(
        params=simple_search_space,
        n_combinations=50,
        random_state=0,
    )


@pytest.fixture
def simple_performances():
    rng = np.random.default_rng(0)
    return list(rng.uniform(0.5, 1.0, size=50).astype(float))


@pytest.fixture
def surrogate_metafeatures_schema():
    return SurrogateMetafeaturesSchema()


@pytest.fixture
def mock_experiment_config(small_param_space):
    generator = BlackBoxGenerator(generator="rastrigin")
    tuner_config = TunerConfig(
        tuner=CustomGPModel(backend="gp_opt", searcher="EI"),
        tuner_identifier="GP-EI",
    )
    return ExperimentConfig(
        search_space=small_param_space,
        objective_function=generator,
        tuner_configurations=[tuner_config],
        benchmark_identifier="blackbox",
        dataset_identifier="rastrigin",
    )


@pytest.fixture
def non_confopt_tuner():
    return TunerConfig(
        tuner=CustomGPModel(backend="gp_opt", searcher="EI"),
        tuner_identifier="GP-EI",
    )


@pytest.fixture
def trial_row():
    return pd.DataFrame({"performance": [0.9], "configurations": [{"x": 0.5}], "iteration": [1]})


@pytest.fixture
def aliases():
    return Aliases()


@pytest.fixture
def blackbox_experiment_config(small_param_space):
    return ExperimentConfig(
        search_space=small_param_space,
        objective_function=BlackBoxGenerator(generator="rastrigin"),
        tuner_configurations=[],
        benchmark_identifier="blackbox",
        dataset_identifier="rastrigin",
    )


@pytest.fixture
def benchmark_data_schema():
    return BenchmarkDataSchema()


@pytest.fixture
def base_random_state():
    return 42


@pytest.fixture
def ltr_train_data():
    """Create toy training data for LTR models.
    
    Structure: 4 groups, 3 tuners, deterministic performance ranking
    based on feature values. tuner_a is always best, tuner_c is second, tuner_b is worst.
    """
    np.random.seed(42)
    data_list = []
    
    for group_id in range(4):
        for tuner_name, base_feature_offset in [('tuner_a', 0), ('tuner_b', 20), ('tuner_c', 10)]:
            rank_value = 1 if tuner_name == 'tuner_a' else (3 if tuner_name == 'tuner_b' else 2)
            data_list.append({
                'ranking_group': group_id,
                'tuner': tuner_name,
                'label': rank_value,
                'feature_1': 10.0 + base_feature_offset + np.random.normal(0, 0.5),
                'feature_2': 2.0 + base_feature_offset * 0.05 + np.random.normal(0, 0.1),
            })
    
    return pd.DataFrame(data_list)


@pytest.fixture
def ltr_test_data():
    """Create toy test data for LTR models with same deterministic structure."""
    np.random.seed(43)
    data_list = []
    
    for group_id in range(4, 6):
        for tuner_name, base_feature_offset in [('tuner_a', 0), ('tuner_b', 20), ('tuner_c', 10)]:
            data_list.append({
                'ranking_group': group_id,
                'tuner': tuner_name,
                'feature_1': 11.0 + base_feature_offset + np.random.normal(0, 0.5),
                'feature_2': 2.1 + base_feature_offset * 0.05 + np.random.normal(0, 0.1),
            })
    
    return pd.DataFrame(data_list)


@pytest.fixture
def ltr_deterministic_simple_train():
    """Create extremely simple deterministic training data with clear feature separation.
    
    Each tuner has exactly same features across all groups and a consistent rank:
    - tuner_a: features=[100, 100], rank=1 (best)
    - tuner_b: features=[10, 10], rank=3 (worst)
    - tuner_c: features=[50, 50], rank=2 (middle)
    
    This should allow the model to learn a clear pattern.
    """
    data_list = []
    tuner_configs = {
        'tuner_a': {'features': [100.0, 100.0], 'rank': 1},
        'tuner_b': {'features': [10.0, 10.0], 'rank': 3},
        'tuner_c': {'features': [50.0, 50.0], 'rank': 2},
    }
    
    for group_id in range(5):
        for tuner_name, config in tuner_configs.items():
            data_list.append({
                'ranking_group': group_id,
                'tuner': tuner_name,
                'label': config['rank'],
                'feature_1': config['features'][0],
                'feature_2': config['features'][1],
            })
    
    return pd.DataFrame(data_list)


@pytest.fixture
def ltr_deterministic_simple_test():
    """Create simple deterministic test data matching training structure for validation."""
    data_list = []
    tuner_configs = {
        'tuner_a': {'features': [100.0, 100.0]},
        'tuner_b': {'features': [10.0, 10.0]},
        'tuner_c': {'features': [50.0, 50.0]},
    }
    
    for group_id in range(5, 8):
        for tuner_name, config in tuner_configs.items():
            data_list.append({
                'ranking_group': group_id,
                'tuner': tuner_name,
                'feature_1': config['features'][0],
                'feature_2': config['features'][1],
            })
    
    return pd.DataFrame(data_list)


@pytest.fixture
def preprocessing_raw_data():
    """Raw benchmark data with deterministic performance for testing preprocessing pipeline."""
    data_list = []
    
    for dataset in ['dataset_a', 'dataset_b']:
        for warm_start_strategy in ['random', 'sobol']:
            for n_warm_starts in [5, 10]:
                for rep in range(2):
                    tuner_perfs = {'tuner_x': 0.1, 'tuner_y': 0.5, 'tuner_z': 0.9}
                    for tuner, perf in tuner_perfs.items():
                        data_list.append({
                            'dataset': dataset,
                            'warm_start_strategy': warm_start_strategy,
                            'n_random_warm_starts': n_warm_starts,
                            'repetition': rep,
                            'tuner': tuner,
                            'performance': perf + rep * 0.01,
                            'benchmark_identifier': 'synthetic_tabular' if dataset == 'dataset_a' else 'lcbench',
                            'n_hyperparameters': 5,
                            'performance_mean': 0.5,
                            'performance_std': 0.2,
                        })
    
    return pd.DataFrame(data_list)

@pytest.fixture
def ltr_config():
    return LTRConfig(
        train_size=0.6,
        val_size=0.2,
        random_state=42,
        k_values=(1, 3),
        tuning=LTRTuningConfig(n_tuning_trials=1)
    )

@pytest.fixture
def ltr_analysis(benchmark_data_schema, surrogate_metafeatures_schema, ltr_config):
    from hpobench.learning_to_rank.analysis import LTRAnalysis
    return LTRAnalysis(
        schema=benchmark_data_schema,
        metafeatures_schema=surrogate_metafeatures_schema,
        synthetic_benchmark_id="synthetic_tabular",
        ltr_config=ltr_config,
        partition="all",
        strategy="random",
        tuner_encoding_method="ordinal"
    )

@pytest.fixture
def fitted_ltr_analysis(ltr_analysis, preprocessing_raw_data):
    ltr_analysis.fit(preprocessing_raw_data, k_values=(1, 3))
    return ltr_analysis
