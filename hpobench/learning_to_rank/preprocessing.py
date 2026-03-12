import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

from hpobench.config.schema import BenchmarkDataSchema, SurrogateMetafeaturesSchema
from hpobench.config.types import Partition, SplitStrategy, TunerEncoding


def _compute_ranks(
    data: pd.DataFrame,
    rank_group_cols: list[str],
    performance_col: str,
    label_col: str,
) -> pd.DataFrame:
    """Compute ranking labels within groups based on performance.
    
    Ranks tuners within each unique combination of grouping columns based on their
    performance values. Lower performance values receive better (lower) ranks.
    
    Args:
        data: Benchmark trial results.
        rank_group_cols: Columns defining unique ranking contexts.
        performance_col: Column containing performance metric values.
        label_col: Name for output ranking label column.
    
    Returns:
        Copy of data with added label_col containing within-group ranks.
    """
    result = data.copy()
    result[label_col] = (
        result.groupby(rank_group_cols, group_keys=False)[performance_col]
        .rank(method='average', ascending=True)
    )
    return result


def _add_engineered_columns(
    data: pd.DataFrame,
    rank_group_cols: list[str],
    split_group_cols: list[str],
    ranking_group_id_col: str,
    split_group_id_col: str,
) -> list[str]:
    """Create concatenated identifier columns for ranking and data splitting.
    
    Creates two engineered columns by concatenating source columns:
    - ranking_group_id_col: All rank_group_cols concatenated
    - split_group_id_col: All split_group_cols concatenated
    
    Args:
        data: Benchmark data (modified in place).
        rank_group_cols: Columns for ranking groups.
        split_group_cols: Columns for split groups.
        ranking_group_id_col: Name for ranking group identifier column.
        split_group_id_col: Name for split group identifier column.
    
    Returns:
        List of newly created column names.
    """
    data[ranking_group_id_col] = data[rank_group_cols].astype(str).agg('_'.join, axis=1)
    data[split_group_id_col] = data[split_group_cols].astype(str).agg('_'.join, axis=1)
    return [ranking_group_id_col, split_group_id_col]


def _encode_tuner(
    data: pd.DataFrame,
    tuner_col: str,
    method: TunerEncoding,
) -> list[str]:
    """Encode tuner identities as numeric or one-hot features.
    
    Args:
        data: Benchmark data (modified in place).
        tuner_col: Column containing tuner identifiers.
        method: Encoding method ('ordinal' or 'one_hot').
    
    Returns:
        List of newly created encoded column names.
    
    Raises:
        ValueError: If method is not 'ordinal' or 'one_hot'.
    """
    if method == 'ordinal':
        data['tuner_encoded'] = LabelEncoder().fit_transform(data[tuner_col])
        return ['tuner_encoded']
    elif method == 'one_hot':
        dummies = pd.get_dummies(data[tuner_col], prefix='tuner', drop_first=False)
        for col in dummies.columns:
            data[col] = dummies[col]
        return list(dummies.columns)
    raise ValueError(f"Unknown tuner_encoding_method '{method}'. Must be 'ordinal' or 'one_hot'.")


def prepare_data(
    raw_data: pd.DataFrame,
    partition: Partition,
    schema: BenchmarkDataSchema,
    metafeatures_schema: SurrogateMetafeaturesSchema,
    synthetic_benchmark_id: str,
    tuner_encoding_method: TunerEncoding = 'ordinal',
) -> tuple[pd.DataFrame, list[str]]:
    """Prepare raw benchmark data for learning-to-rank model training.
    
    Applies preprocessing pipeline: partition filtering → rank computation → 
    engineered columns → tuner encoding → column subsetting.
    
    Args:
        raw_data: Unfiltered benchmark trial results.
        partition: Data partition ('synthetic', 'real', or 'all').
        schema: Column name schema for benchmark data.
        metafeatures_schema: Feature column schema.
        synthetic_benchmark_id: Identifier for synthetic benchmark rows.
        tuner_encoding_method: Tuner encoding ('ordinal' or 'one_hot').
    
    Returns:
        Tuple of (prepared_data, feature_column_names).
    
    Raises:
        ValueError: If partition is invalid or yields no data.
    """
    if partition == 'synthetic':
        data = raw_data[raw_data[schema.benchmark_identifier_col] == synthetic_benchmark_id].copy()
    elif partition == 'real':
        data = raw_data[raw_data[schema.benchmark_identifier_col] != synthetic_benchmark_id].copy()
    elif partition == 'all':
        data = raw_data.copy()
    else:
        raise ValueError(f"Unknown partition '{partition}'. Must be 'synthetic', 'real', or 'all'.")

    if data.empty:
        raise ValueError(f"No data available for partition '{partition}'")
    
    data = _compute_ranks(
        data=data,
        rank_group_cols=schema.rank_group_cols,
        performance_col=schema.performance_col,
        label_col=schema.label_col,
    )
    
    engineered_cols = _add_engineered_columns(
        data=data,
        rank_group_cols=schema.rank_group_cols,
        split_group_cols=schema.split_group_cols,
        ranking_group_id_col=schema.ranking_group_id_col,
        split_group_id_col=schema.split_group_id_col,
    )
    
    encoded_cols = _encode_tuner(
        data=data,
        tuner_col=schema.tuner_col,
        method=tuner_encoding_method,
    )
    
    available_features = [c for c in metafeatures_schema.to_list() if c in data.columns]
    feature_cols = available_features + encoded_cols
    retained_cols = feature_cols + engineered_cols + [schema.label_col, schema.benchmark_identifier_col, schema.tuner_col]
    data = data[retained_cols]
    
    return data, feature_cols


def split_data(
    data: pd.DataFrame,
    strategy: SplitStrategy,
    train_size: float,
    val_size: float,
    random_state: int,
    synthetic_benchmark_id: str,
    schema: BenchmarkDataSchema,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split prepared data into train/validation/test sets with group integrity.
    
    Maintains group integrity during splitting to prevent data leakage. Groups are
    defined by split_group_id_col (dataset × warm_start_strategy × n_warm_starts).
    
    Args:
        data: Output from prepare_data().
        strategy: Split strategy ('random' or 'synthetic_train_real_test').
        train_size: Training set proportion (0-1).
        val_size: Validation set proportion (0-1).
        random_state: Random seed for reproducibility.
        synthetic_benchmark_id: Identifier for synthetic benchmark rows.
        schema: Column name schema for benchmark data.
    
    Returns:
        Tuple of (train_data, val_data, test_data).
    
    Raises:
        ValueError: If strategy is invalid or synthetic data missing when required.
    """
    val_prop = val_size / (train_size + val_size)

    if strategy == 'synthetic_train_real_test':
        is_synthetic = data[schema.benchmark_identifier_col] == synthetic_benchmark_id
        synthetic, real = data[is_synthetic], data[~is_synthetic]
        if synthetic.empty:
            raise ValueError("No synthetic data available for training")
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_prop, random_state=random_state)
        train_idx, val_idx = next(splitter.split(synthetic, groups=synthetic[schema.split_group_id_col]))
        return synthetic.iloc[train_idx], synthetic.iloc[val_idx], real

    elif strategy == 'random':
        outer = GroupShuffleSplit(
            n_splits=1,
            train_size=train_size + val_size,
            random_state=random_state,
        )
        train_val_idx, test_idx = next(outer.split(data, groups=data[schema.split_group_id_col]))
        train_val, test_data = data.iloc[train_val_idx], data.iloc[test_idx]

        inner = GroupShuffleSplit(n_splits=1, test_size=val_prop, random_state=random_state)
        train_idx, val_idx = next(inner.split(train_val, groups=train_val[schema.split_group_id_col]))
        return train_val.iloc[train_idx], train_val.iloc[val_idx], test_data
    
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Must be 'random' or 'synthetic_train_real_test'.")
