import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

from hpobench.config.schema import BenchmarkDataSchema, SurrogateMetafeaturesSchema
from hpobench.config.types import LTRConfig, Partition, SplitStrategy, TunerEncoding



def _filter_partition(
    raw_data: pd.DataFrame,
    partition: Partition,
    synthetic_benchmark_id: str,
    benchmark_id_col: str = 'benchmark_identifier',
) -> pd.DataFrame:
    if partition == 'synthetic':
        data = raw_data[raw_data[benchmark_id_col] == synthetic_benchmark_id].copy()
    elif partition == 'real':
        data = raw_data[raw_data[benchmark_id_col] != synthetic_benchmark_id].copy()
    elif partition == 'all':
        data = raw_data.copy()
    else:
        raise ValueError(f"Unknown partition '{partition}'. Must be 'synthetic', 'real', or 'all'.")
    if data.empty:
        raise ValueError(f"No data available for partition '{partition}'")
    return data


def _compute_labels(data: pd.DataFrame, schema: BenchmarkDataSchema) -> pd.DataFrame:
    group_cols = [schema.data_col, schema.warm_start_strategy_col, schema.n_random_warm_starts_col, schema.rep_col]
    data = data.copy()
    data[schema.label_col] = (
        data.groupby(group_cols, group_keys=False)[schema.performance_col]
        .rank(method='average', ascending=True)
    )
    return data


def _select_features(
    data: pd.DataFrame,
    metafeatures_schema: SurrogateMetafeaturesSchema,
) -> tuple[pd.DataFrame, list[str]]:
    known = set(metafeatures_schema.to_list())
    feature_cols = [c for c in data.columns if c in known]
    return data, feature_cols


def _organize_columns(
    data: pd.DataFrame,
    schema: BenchmarkDataSchema,
    feature_cols: list[str],
    benchmark_id_col: str = 'benchmark_identifier',
) -> pd.DataFrame:
    base_cols = [
        schema.data_col, schema.warm_start_strategy_col, schema.n_random_warm_starts_col,
        schema.rep_col, schema.tuner_col, schema.label_col, benchmark_id_col,
    ]
    base_cols = [c for c in base_cols if c in data.columns]
    return data[base_cols + [c for c in feature_cols if c not in base_cols]].copy()


def _encode_tuner(
    data: pd.DataFrame,
    feature_cols: list[str],
    schema: BenchmarkDataSchema,
    method: TunerEncoding,
) -> tuple[pd.DataFrame, list[str]]:
    data = data.copy()
    if method == 'ordinal':
        data['tuner_encoded'] = LabelEncoder().fit_transform(data[schema.tuner_col])
        return data, feature_cols + ['tuner_encoded']
    elif method == 'one_hot':
        dummies = pd.get_dummies(data[schema.tuner_col], prefix='tuner', drop_first=False)
        data = pd.concat([data, dummies], axis=1)
        return data, feature_cols + list(dummies.columns)
    raise ValueError(f"Unknown tuner_encoding_method '{method}'. Must be 'ordinal' or 'one_hot'.")


def _add_grouping_columns(
    data: pd.DataFrame,
    feature_cols: list[str],
    schema: BenchmarkDataSchema,
) -> tuple[pd.DataFrame, list[str]]:
    data = data.copy()
    ws_col = schema.n_random_warm_starts_col
    strat_col = schema.warm_start_strategy_col
    if ws_col in data.columns and ws_col not in feature_cols:
        feature_cols = feature_cols + [ws_col]
    data[schema.ranking_group_col] = (
        data[schema.data_col].astype(str) + '_'
        + data[strat_col].astype(str) + '_'
        + data[ws_col].astype(str) + '_'
        + data[schema.rep_col].astype(str)
    )
    data['split_group'] = (
        data[schema.data_col].astype(str) + '_'
        + data[strat_col].astype(str) + '_'
        + data[ws_col].astype(str)
    )
    return data, feature_cols


def prepare_data(
    raw_data: pd.DataFrame,
    partition: Partition,
    schema: BenchmarkDataSchema,
    metafeatures_schema: SurrogateMetafeaturesSchema,
    synthetic_benchmark_id: str,
    tuner_encoding_method: TunerEncoding = 'ordinal',
) -> tuple[pd.DataFrame, list[str]]:
    """Prepare raw benchmark data for learning-to-rank training.

    Applies the full preprocessing pipeline in order:
    partition filtering → label computation → feature selection →
    column organisation → tuner encoding → grouping columns.

    Returns:
        ``(prepared_df, feature_cols)``

    Raises:
        ValueError: If the requested partition yields no data.
    """
    data = _filter_partition(raw_data, partition, synthetic_benchmark_id)
    data = _compute_labels(data, schema)
    data, feature_cols = _select_features(data, metafeatures_schema)
    data = _organize_columns(data, schema, feature_cols)
    data, feature_cols = _encode_tuner(data, feature_cols, schema, tuner_encoding_method)
    data, feature_cols = _add_grouping_columns(data, feature_cols, schema)
    return data, feature_cols


def split_data(
    data: pd.DataFrame,
    strategy: SplitStrategy,
    config: LTRConfig,
    synthetic_benchmark_id: str,
    benchmark_id_col: str = 'benchmark_identifier',
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split prepared data into train / validation / test sets.

    Groups (dataset × n_warm_starts combinations) are kept intact so no
    dataset leaks across splits.

    Args:
        data: Output of :func:`prepare_data`.
        strategy:
            ``'random'`` – group-stratified random split.
            ``'synthetic_train_real_test'`` – train/val on synthetic rows,
            test on real rows.
        config: Provides ``train_size``, ``val_size``, and ``random_state``.
        synthetic_benchmark_id: Identifier value for synthetic benchmark rows.
        benchmark_id_col: Column name holding the benchmark identifier.

    Returns:
        ``(train_data, val_data, test_data)``

    Raises:
        ValueError: Strategy is ``'synthetic_train_real_test'`` but no
            synthetic rows exist.
    """
    val_prop = config.val_size / (config.train_size + config.val_size)

    if strategy == 'synthetic_train_real_test':
        is_synthetic = data[benchmark_id_col] == synthetic_benchmark_id
        synthetic, real = data[is_synthetic], data[~is_synthetic]
        if synthetic.empty:
            raise ValueError("No synthetic data available for training")
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_prop, random_state=config.random_state)
        train_idx, val_idx = next(splitter.split(synthetic, groups=synthetic['split_group']))
        return synthetic.iloc[train_idx], synthetic.iloc[val_idx], real

    outer = GroupShuffleSplit(
        n_splits=1, train_size=config.train_size + config.val_size, random_state=config.random_state
    )
    train_val_idx, test_idx = next(outer.split(data, groups=data['split_group']))
    train_val, test_data = data.iloc[train_val_idx], data.iloc[test_idx]

    inner = GroupShuffleSplit(n_splits=1, test_size=val_prop, random_state=config.random_state)
    train_idx, val_idx = next(inner.split(train_val, groups=train_val['split_group']))
    return train_val.iloc[train_idx], train_val.iloc[val_idx], test_data

