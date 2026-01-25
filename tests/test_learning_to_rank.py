import pytest
import pandas as pd
import numpy as np
from hpobench.report.learning_to_rank import (
    prepare_ranking_data,
    split_ranking_groups,
    train_naive_ranker,
    train_ltr_model,
    evaluate_ltr_model,
    evaluate_naive_ranker,
    calculate_precision_at_k,
    calculate_ndcg_at_k,
    run_learning_to_rank_analysis,
)


@pytest.fixture
def sample_raw_benchmark_data():
    """Create sample raw benchmark data for testing."""
    np.random.seed(42)
    
    datasets = ['dataset_1', 'dataset_2', 'dataset_3']
    tuners = ['tuner_A', 'tuner_B', 'tuner_C']
    repetitions = [1, 2]
    trials = list(range(1, 11))
    
    rows = []
    for dataset in datasets:
        for repetition in repetitions:
            for trial in trials:
                for tuner in tuners:
                    performance = np.random.uniform(0.5, 0.95)
                    row = {
                        'trial': trial,
                        'performance': performance,
                        'runtime': np.random.uniform(1, 10),
                        'benchmark_identifier': 'test_benchmark',
                        'dataset': dataset,
                        'tuner': tuner,
                        'repetition': repetition,
                        'searcher_tuning_framework': 'test_framework',
                        'estimator_architecture': '',
                        'confidence_level': '',
                        'sampler': '',
                        'n_pre_conformal_trials': '',
                        'sampler_n_quantiles': '',
                        'sampler_adapter': '',
                        'tuner_searcher_tuning_framework': '',
                        'n_integer_hyperparameters': 2,
                        'n_float_hyperparameters': 1,
                        'n_categorical_hyperparameters': 1,
                        'ratio_continuous_hyperparameters': 0.75,
                        'ratio_categorical_hyperparameters': 0.25,
                        'avg_categorical_cardinality': 3.0,
                        'min_categorical_cardinality': 3.0,
                        'max_categorical_cardinality': 3.0,
                        'total_search_space_combinations': 1000.0,
                        'n_samples': np.random.randint(100, 1000),
                        'n_features': np.random.randint(10, 50),
                    }
                    rows.append(row)
    
    return pd.DataFrame(rows)


def test_prepare_ranking_data(sample_raw_benchmark_data):
    """Test ranking data preparation."""
    ranking_data = prepare_ranking_data(sample_raw_benchmark_data)
    
    assert 'ranking_group' in ranking_data.columns
    assert 'iteration' in ranking_data.columns
    assert 'trial' not in ranking_data.columns
    
    n_datasets = sample_raw_benchmark_data['dataset'].nunique()
    n_repetitions = sample_raw_benchmark_data['repetition'].nunique()
    n_trials = sample_raw_benchmark_data['trial'].nunique()
    expected_groups = n_datasets * n_repetitions * n_trials
    
    assert ranking_data['ranking_group'].nunique() == expected_groups
    
    assert 'n_integer_hyperparameters' in ranking_data.columns
    assert 'n_samples' in ranking_data.columns


def test_split_ranking_groups(sample_raw_benchmark_data):
    """Test ranking group splitting."""
    ranking_data = prepare_ranking_data(sample_raw_benchmark_data)
    
    train_data, val_data, test_data = split_ranking_groups(
        ranking_data=ranking_data,
        train_size=0.7,
        val_size=0.15,
        random_state=42,
    )
    
    assert len(train_data) > 0
    assert len(val_data) > 0
    assert len(test_data) > 0
    
    total_rows = len(train_data) + len(val_data) + len(test_data)
    assert total_rows == len(ranking_data)
    
    train_groups = set(train_data['ranking_group'].unique())
    val_groups = set(val_data['ranking_group'].unique())
    test_groups = set(test_data['ranking_group'].unique())
    
    assert len(train_groups.intersection(val_groups)) == 0
    assert len(train_groups.intersection(test_groups)) == 0
    assert len(val_groups.intersection(test_groups)) == 0


def test_train_naive_ranker(sample_raw_benchmark_data):
    """Test naive ranker training."""
    ranking_data = prepare_ranking_data(sample_raw_benchmark_data)
    train_data, _, _ = split_ranking_groups(ranking_data, random_state=42)
    
    naive_ranker = train_naive_ranker(train_data)
    
    assert isinstance(naive_ranker, dict)
    assert len(naive_ranker) > 0
    
    for tuner, avg_rank in naive_ranker.items():
        assert isinstance(tuner, str)
        assert isinstance(avg_rank, float)
        assert avg_rank >= 1.0


def test_calculate_precision_at_k():
    """Test precision@k calculation."""
    predicted = ['A', 'B', 'C', 'D']
    true = ['A', 'C', 'E', 'F']
    
    precision_1 = calculate_precision_at_k(predicted, true, k=1)
    assert precision_1 == 1.0
    
    precision_2 = calculate_precision_at_k(predicted, true, k=2)
    assert precision_2 == 0.5
    
    precision_3 = calculate_precision_at_k(predicted, true, k=3)
    assert abs(precision_3 - 2/3) < 1e-6


def test_calculate_ndcg_at_k():
    """Test NDCG@k calculation."""
    predicted_scores = np.array([10, 8, 6, 4])
    true_relevance = np.array([10, 6, 8, 4])
    
    ndcg = calculate_ndcg_at_k(predicted_scores, true_relevance, k=3)
    
    assert 0.0 <= ndcg <= 1.0


def test_train_ltr_model(sample_raw_benchmark_data):
    """Test LTR model training."""
    ranking_data = prepare_ranking_data(sample_raw_benchmark_data)
    train_data, val_data, _ = split_ranking_groups(ranking_data, random_state=42)
    
    feature_cols = [
        col for col in ranking_data.columns 
        if col not in [
            'dataset', 'repetition', 'iteration', 'tuner', 
            'performance', 'ranking_group', 'label'
        ]
    ]
    
    xgb_params = {
        'objective': 'rank:ndcg',
        'learning_rate': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'verbosity': 0,
        'seed': 42,
    }
    
    model = train_ltr_model(
        train_data=train_data,
        val_data=val_data,
        feature_cols=feature_cols,
        xgb_params=xgb_params,
    )
    
    assert model is not None
    assert hasattr(model, 'predict')


def test_evaluate_ltr_model(sample_raw_benchmark_data):
    """Test LTR model evaluation."""
    ranking_data = prepare_ranking_data(sample_raw_benchmark_data)
    train_data, val_data, test_data = split_ranking_groups(
        ranking_data, random_state=42
    )
    
    feature_cols = [
        col for col in ranking_data.columns 
        if col not in [
            'dataset', 'repetition', 'iteration', 'tuner', 
            'performance', 'ranking_group', 'label'
        ]
    ]
    
    xgb_params = {
        'objective': 'rank:ndcg',
        'learning_rate': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'verbosity': 0,
        'seed': 42,
    }
    
    model = train_ltr_model(
        train_data=train_data,
        val_data=val_data,
        feature_cols=feature_cols,
        xgb_params=xgb_params,
    )
    
    metrics = evaluate_ltr_model(
        model=model,
        test_data=test_data,
        feature_cols=feature_cols,
        k_values=[1, 3],
    )
    
    assert 'precision@1' in metrics
    assert 'precision@3' in metrics
    assert 'ndcg@1' in metrics
    assert 'ndcg@3' in metrics
    
    for metric_value in metrics.values():
        assert 0.0 <= metric_value <= 1.0


def test_evaluate_naive_ranker(sample_raw_benchmark_data):
    """Test naive ranker evaluation."""
    ranking_data = prepare_ranking_data(sample_raw_benchmark_data)
    train_data, _, test_data = split_ranking_groups(ranking_data, random_state=42)
    
    naive_ranker = train_naive_ranker(train_data)
    
    metrics = evaluate_naive_ranker(
        naive_ranker=naive_ranker,
        test_data=test_data,
        k_values=[1, 3],
    )
    
    assert 'precision@1' in metrics
    assert 'precision@3' in metrics
    assert 'ndcg@1' in metrics
    assert 'ndcg@3' in metrics
    
    for metric_value in metrics.values():
        assert 0.0 <= metric_value <= 1.0


def test_run_learning_to_rank_analysis(sample_raw_benchmark_data):
    """Test complete learning-to-rank analysis pipeline."""
    xgb_params = {
        'objective': 'rank:ndcg',
        'learning_rate': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'verbosity': 0,
        'seed': 42,
    }
    
    results = run_learning_to_rank_analysis(
        raw_benchmark_data=sample_raw_benchmark_data,
        train_size=0.7,
        val_size=0.15,
        random_state=42,
        k_values=[1, 3],
        xgb_params=xgb_params,
    )
    
    assert 'ltr_model' in results
    assert 'naive_ranker' in results
    assert 'ltr_metrics' in results
    assert 'naive_metrics' in results
    assert 'test_data' in results
    assert 'feature_cols' in results
    
    assert hasattr(results['ltr_model'], 'predict')
    assert isinstance(results['naive_ranker'], dict)
    assert isinstance(results['ltr_metrics'], dict)
    assert isinstance(results['naive_metrics'], dict)
    assert isinstance(results['test_data'], pd.DataFrame)
    assert isinstance(results['feature_cols'], list)
    
    for k in [1, 3]:
        assert f'precision@{k}' in results['ltr_metrics']
        assert f'ndcg@{k}' in results['ltr_metrics']
        assert f'precision@{k}' in results['naive_metrics']
        assert f'ndcg@{k}' in results['naive_metrics']


def test_ranking_data_has_correct_features(sample_raw_benchmark_data):
    """Test that ranking data contains all expected features."""
    ranking_data = prepare_ranking_data(sample_raw_benchmark_data)
    
    expected_features = [
        'n_integer_hyperparameters',
        'n_float_hyperparameters',
        'n_categorical_hyperparameters',
        'ratio_continuous_hyperparameters',
        'ratio_categorical_hyperparameters',
        'avg_categorical_cardinality',
        'min_categorical_cardinality',
        'max_categorical_cardinality',
        'total_search_space_combinations',
        'n_samples',
        'n_features',
        'iteration',
    ]
    
    for feature in expected_features:
        assert feature in ranking_data.columns


def test_ranking_groups_have_multiple_tuners(sample_raw_benchmark_data):
    """Test that each ranking group has multiple tuners to rank."""
    ranking_data = prepare_ranking_data(sample_raw_benchmark_data)
    
    for group_name, group_df in ranking_data.groupby('ranking_group'):
        assert len(group_df) >= 2
        assert group_df['tuner'].nunique() >= 2
