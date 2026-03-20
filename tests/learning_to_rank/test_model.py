import numpy as np
import pandas as pd
import pytest
from hpobench.learning_to_rank.model import AverageRankRanker, LTRModel, _evaluate_rankings
from hpobench.config.types import LTRConfig


def test_average_rank_ranker_predicts_correct_average_ranks(ltr_train_data):
    """AverageRankRanker should output average rank for each tuner across all groups."""
    ranker = AverageRankRanker()
    ranker.fit(
        train_data=ltr_train_data,
        tuner_col='tuner',
        label_col='label',
    )
    
    predictions = ranker.predict(ltr_train_data)
    
    expected_ranks = {
        'tuner_a': 1.0,
        'tuner_b': 3.0,
        'tuner_c': 2.0,
    }
    
    for tuner, expected_rank in expected_ranks.items():
        tuner_indices = ltr_train_data[ltr_train_data['tuner'] == tuner].index
        actual_predictions = predictions[tuner_indices]
        assert np.allclose(actual_predictions, expected_rank)


@pytest.mark.parametrize("n_tuning_trials", [1, 20])
def test_ltr_model_fit_with_and_without_tuning(ltr_deterministic_simple_train, n_tuning_trials):
    """LTRModel.fit() should complete successfully with tuning enabled and disabled.
    
    Tests both n_tuning_trials=1 (tuning disabled) and n_tuning_trials=20 (tuning enabled)
    to ensure the model trains correctly in both scenarios.
    """
    model = LTRModel()
    ltr_config = LTRConfig()
    
    model.fit(
        train_data=ltr_deterministic_simple_train,
        val_data=ltr_deterministic_simple_train[:5],
        group_col='ranking_group',
        label_col='label',
        tuner_col='tuner',
        feature_cols=['feature_1', 'feature_2'],
        tuning_config=ltr_config.tuning,
        k_values=ltr_config.k_values,
        analysis_identifier='test_analysis',
        n_tuning_trials=n_tuning_trials,
    )
    
    assert model.booster is not None
    assert len(model.feature_cols) == 2


@pytest.mark.parametrize("n_tuning_trials", [1, 20])
def test_ltr_model_ranks_consistently(ltr_deterministic_simple_train, n_tuning_trials):
    """LTRModel should learn deterministic ranking patterns with tuning enabled/disabled.
    
    Training data has clear separation: tuner_a has features [100, 100] (best),
    tuner_c has [50, 50] (middle), tuner_b has [10, 10] (worst).
    Model should consistently learn this ranking pattern regardless of tuning setting.
    """
    model = LTRModel()
    ltr_config = LTRConfig()
    
    model.fit(
        train_data=ltr_deterministic_simple_train,
        val_data=ltr_deterministic_simple_train[:5],
        group_col='ranking_group',
        label_col='label',
        tuner_col='tuner',
        feature_cols=['feature_1', 'feature_2'],
        tuning_config=ltr_config.tuning,
        k_values=ltr_config.k_values,
        analysis_identifier='test_model',
        n_tuning_trials=n_tuning_trials,
    )
    
    predictions = model.predict(ltr_deterministic_simple_train)
    
    for group_id in ltr_deterministic_simple_train['ranking_group'].unique():
        group_data = ltr_deterministic_simple_train[ltr_deterministic_simple_train['ranking_group'] == group_id]
        group_indices = group_data.index.tolist()
        
        if len(group_indices) >= 3:
            tuner_a_idx = [idx for idx in group_indices if group_data.loc[idx, 'tuner'] == 'tuner_a'][0]
            tuner_b_idx = [idx for idx in group_indices if group_data.loc[idx, 'tuner'] == 'tuner_b'][0]
            tuner_c_idx = [idx for idx in group_indices if group_data.loc[idx, 'tuner'] == 'tuner_c'][0]
            
            score_a = predictions[tuner_a_idx]
            score_b = predictions[tuner_b_idx]
            score_c = predictions[tuner_c_idx]
            
            assert score_a < score_c < score_b, \
                f"Expected a < c < b, but got a={score_a:.4f}, c={score_c:.4f}, b={score_b:.4f}"


def test_ltr_model_predict_outputs_correct_shape(ltr_deterministic_simple_train):
    """LTRModel.predict() should output predictions with matching shape."""
    model = LTRModel()
    ltr_config = LTRConfig()
    
    model.fit(
        train_data=ltr_deterministic_simple_train,
        val_data=ltr_deterministic_simple_train[:5],
        group_col='ranking_group',
        label_col='label',
        tuner_col='tuner',
        feature_cols=['feature_1', 'feature_2'],
        tuning_config=ltr_config.tuning,
        k_values=ltr_config.k_values,
        analysis_identifier='test_model',
        n_tuning_trials=1,
    )
    
    predictions = model.predict(ltr_deterministic_simple_train)
    assert len(predictions) == len(ltr_deterministic_simple_train)
    assert not np.isnan(predictions).any()


def test_evaluate_rankings_computes_precision_and_ndcg(ltr_test_data):
    """_evaluate_rankings should compute precision@k and NDCG@k metrics."""
    scores = np.random.randn(len(ltr_test_data))
    
    metrics = _evaluate_rankings(
        test_data=ltr_test_data,
        predicted_scores=scores,
        k_values=(1, 3),
        ranking_group_id_col='ranking_group',
        label_col='tuner',
        tuner_col='tuner',
    )
    
    assert 'precision@1' in metrics and 'precision@3' in metrics
    assert 'ndcg@1' in metrics and 'ndcg@3' in metrics
    assert all(0 <= v <= 1 for v in metrics.values())

