import numpy as np
import pandas as pd
import pytest
from hpobench.learning_to_rank.model import NaiveRanker, LTRModel


def test_naive_ranker_predicts_correct_average_ranks(ltr_train_data):
    """NaiveRanker should output average rank for each tuner across all groups."""
    ranker = NaiveRanker()
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


def test_ltr_model_ranks_consistently(ltr_train_data, ltr_test_data):
    """LTRModel should learn clear ranking patterns and predict consistent relative ranks on holdout set."""
    model = LTRModel(
        num_boost_rounds=100,
        objective="rank:ndcg",
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        verbosity=0,
        seed=42,
    )
    
    model.fit(
        train_data=ltr_train_data,
        group_col='ranking_group',
        label_col='label',
        feature_cols=['feature_1', 'feature_2'],
    )
    
    predictions = model.predict(ltr_test_data)
    
    for group_id in ltr_test_data['ranking_group'].unique():
        group_data = ltr_test_data[ltr_test_data['ranking_group'] == group_id]
        group_preds = predictions[group_data.index]
        
        tuner_a_score = group_preds[group_data['tuner'] == 'tuner_a'].item()
        tuner_b_score = group_preds[group_data['tuner'] == 'tuner_b'].item()
        tuner_c_score = group_preds[group_data['tuner'] == 'tuner_c'].item()
        
        assert tuner_a_score < tuner_c_score < tuner_b_score


def test_ltr_model_deterministic_ranking(ltr_deterministic_simple_train, ltr_deterministic_simple_test):
    """LTRModel should learn and predict deterministic ranking patterns exactly on simple synthetic data.
    
    Training data has extremely clear separation: tuner_a has feature values [100, 100],
    tuner_b has [10, 10], and tuner_c has [50, 50]. This clear separation should allow
    the model to learn that higher features correspond to better ranks.
    """
    model = LTRModel(
        num_boost_rounds=200,
        objective="rank:ndcg",
        learning_rate=0.2,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        verbosity=0,
        seed=42,
    )
    
    model.fit(
        train_data=ltr_deterministic_simple_train,
        group_col='ranking_group',
        label_col='label',
        feature_cols=['feature_1', 'feature_2'],
    )
    
    predictions = model.predict(ltr_deterministic_simple_test)
    
    for group_id in ltr_deterministic_simple_test['ranking_group'].unique():
        group_data = ltr_deterministic_simple_test[ltr_deterministic_simple_test['ranking_group'] == group_id]
        group_indices = group_data.index
        
        tuner_a_idx = group_indices[group_data['tuner'] == 'tuner_a'].item()
        tuner_b_idx = group_indices[group_data['tuner'] == 'tuner_b'].item()
        tuner_c_idx = group_indices[group_data['tuner'] == 'tuner_c'].item()
        
        score_a = predictions[tuner_a_idx]
        score_b = predictions[tuner_b_idx]
        score_c = predictions[tuner_c_idx]
        
        assert score_a < score_c < score_b

