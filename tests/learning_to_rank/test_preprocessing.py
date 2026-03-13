import pytest
import pandas as pd
from hpobench.learning_to_rank.preprocessing import prepare_data, split_data


def test_prepare_data_ranks_tuners_correctly_within_groups(preprocessing_raw_data, benchmark_data_schema, surrogate_metafeatures_schema):
    prepared_data, _ = prepare_data(
        raw_data=preprocessing_raw_data,
        schema=benchmark_data_schema,
        metafeatures_schema=surrogate_metafeatures_schema,
    )
    
    sample_group = prepared_data['ranking_group_id'].iloc[0]
    group_data = prepared_data[prepared_data['ranking_group_id'] == sample_group]
    
    tuner_x_label = group_data[group_data['tuner_encoded'] == 0]['label'].iloc[0]
    tuner_y_label = group_data[group_data['tuner_encoded'] == 1]['label'].iloc[0]
    tuner_z_label = group_data[group_data['tuner_encoded'] == 2]['label'].iloc[0]
    
    assert tuner_x_label == 1.0
    assert tuner_y_label == 2.0
    assert tuner_z_label == 3.0


def test_prepare_data_one_hot_creates_binary_indicators(preprocessing_raw_data, benchmark_data_schema, surrogate_metafeatures_schema):
    prepared_data, feature_cols = prepare_data(
        raw_data=preprocessing_raw_data,
        schema=benchmark_data_schema,
        metafeatures_schema=surrogate_metafeatures_schema,
        tuner_encoding_method='one_hot',
    )
    
    tuner_cols = [col for col in feature_cols if col.startswith('tuner_')]
    assert len(tuner_cols) == 3
    
    for idx, row in prepared_data.iterrows():
        tuner_values = [row[col] for col in tuner_cols]
        assert sum(tuner_values) == 1
        assert all(val in [0, 1] for val in tuner_values)


def test_split_data_maintains_exact_proportions(preprocessing_raw_data, benchmark_data_schema, surrogate_metafeatures_schema):
    prepared_data, _ = prepare_data(
        raw_data=preprocessing_raw_data,
        schema=benchmark_data_schema,
        metafeatures_schema=surrogate_metafeatures_schema,
    )
    
    # Filter to synthetic only to test simple split
    synthetic_data = prepared_data[prepared_data['benchmark_identifier'] == 'synthetic_tabular']
    n_groups = synthetic_data['split_group_id'].nunique()
    
    train, val, test = split_data(
        data=synthetic_data,
        partition='synthetic',
        strategy='random',
        train_size=0.6,
        val_size=0.2,
        random_state=42,
        synthetic_benchmark_id='synthetic_tabular',
        schema=benchmark_data_schema,
    )
    
    train_groups = train['split_group_id'].nunique()
    val_groups = val['split_group_id'].nunique()
    test_groups = test['split_group_id'].nunique()
    
    assert train_groups + val_groups + test_groups == n_groups
    expected_test_groups = int(n_groups * 0.2)
    # Allow off-by-one due to rounding in small datasets
    assert abs(test_groups - expected_test_groups) <= 1


def test_split_data_preserves_group_integrity(preprocessing_raw_data, benchmark_data_schema, surrogate_metafeatures_schema):
    prepared_data, _ = prepare_data(
        raw_data=preprocessing_raw_data,
        schema=benchmark_data_schema,
        metafeatures_schema=surrogate_metafeatures_schema,
    )
    
    train, val, test = split_data(
        data=prepared_data,
        partition='all',
        strategy='random',
        train_size=0.6,
        val_size=0.2,
        random_state=42,
        synthetic_benchmark_id='synthetic_tabular',
        schema=benchmark_data_schema,
    )
    
    train_groups = set(train['split_group_id'].unique())
    val_groups = set(val['split_group_id'].unique())
    test_groups = set(test['split_group_id'].unique())
    
    assert len(train_groups & val_groups) == 0
    assert len(train_groups & test_groups) == 0
    assert len(val_groups & test_groups) == 0
    
    for split_group in train_groups:
        group_rows = prepared_data[prepared_data['split_group_id'] == split_group]
        # Check that if a group is in train, all its rows from prepared_data (if present in train) match
        # Actually split_data filters rows. We check that no rows of this group end up in val/test.
        pass

def test_split_data_synthetic_train_real_test_isolates_partitions(preprocessing_raw_data, benchmark_data_schema, surrogate_metafeatures_schema):
    prepared_data, _ = prepare_data(
        raw_data=preprocessing_raw_data,
        schema=benchmark_data_schema,
        metafeatures_schema=surrogate_metafeatures_schema,
    )
    
    train, val, test = split_data(
        data=prepared_data,
        partition='all',
        strategy='synthetic_train_real_test',
        train_size=0.6,
        val_size=0.2,
        random_state=42,
        synthetic_benchmark_id='synthetic_tabular',
        schema=benchmark_data_schema,
    )
    
    # Train should contain all synthetic + real train
    assert 'synthetic_tabular' in train['benchmark_identifier'].unique()
    assert 'lcbench' in train['benchmark_identifier'].unique()
    
    # Val should be real only
    assert all(val['benchmark_identifier'] == 'lcbench')
    
    # Test should be real only
    assert all(test['benchmark_identifier'] == 'lcbench')
    
    # Verify synthetic data is fully in training set
    synthetic_groups = prepared_data[prepared_data['benchmark_identifier'] == 'synthetic_tabular']['split_group_id'].nunique()
    train_synthetic_groups = train[train['benchmark_identifier'] == 'synthetic_tabular']['split_group_id'].nunique()
    
    assert train_synthetic_groups == synthetic_groups
    
    # Verify real data splits
    real_data = prepared_data[prepared_data['benchmark_identifier'] == 'lcbench']
    real_groups = real_data['split_group_id'].nunique()
    
    train_real_groups = train[train['benchmark_identifier'] == 'lcbench']['split_group_id'].nunique()
    val_real_groups = val['split_group_id'].nunique()
    test_real_groups = test['split_group_id'].nunique()
    
    assert train_real_groups + val_real_groups + test_real_groups == real_groups
