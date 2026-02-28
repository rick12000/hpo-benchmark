"""
Data preparation for learning-to-rank: feature engineering and train/val/test splits.

Deliberately free of model logic – only transforms raw benchmark data into the
format expected by LTR training and partitions it for train/val/test.
"""

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from typing import Literal

from hpobench.config.schema import BenchmarkDataSchema, SurrogateMetafeaturesSchema
from hpobench.config.constants import SyntheticGenerationParameters
from hpobench.learning_to_rank.model import LTRConfig

SYNTHETIC_BENCHMARK: str = SyntheticGenerationParameters().benchmark_identifier
BENCHMARK_ID_COL: str = 'benchmark_identifier'


def prepare_data(
    raw_data: pd.DataFrame,
    schema: BenchmarkDataSchema,
    partition: Literal['all', 'synthetic', 'real'] = 'all',
    tuner_encoding_method: Literal['ordinal', 'one_hot'] = 'ordinal',
) -> tuple[pd.DataFrame, list[str]]:
    """Prepare raw benchmark data for learning-to-rank.

    Steps:
    1. Filter to the requested partition.
    2. Compute within-group performance ranks as LTR labels.
    3. Encode algorithm identity as a feature.
    4. Add grouping columns used by :func:`split_data`.

    Args:
        raw_data: Raw benchmark experiment data.
        schema: Column name schema.
        partition: Data subset – ``'all'``, ``'synthetic'``, or ``'real'``.
        tuner_encoding_method:
            ``'ordinal'`` – single integer label.
            ``'one_hot'`` – binary indicator per tuner.

    Returns:
        ``(prepared_df, feature_cols)``

    Raises:
        ValueError: Empty partition or unknown *tuner_encoding_method*.
    """
    if partition == 'synthetic':
        data = raw_data[raw_data[BENCHMARK_ID_COL] == SYNTHETIC_BENCHMARK].copy()
    elif partition == 'real':
        data = raw_data[raw_data[BENCHMARK_ID_COL] != SYNTHETIC_BENCHMARK].copy()
    else:
        data = raw_data.copy()

    if data.empty:
        raise ValueError(f"No data available for partition '{partition}'")

    group_cols = [schema.data_col, schema.rep_col, schema.n_random_warm_starts_col]
    data[schema.label_col] = (
        data.groupby(group_cols, group_keys=False)[schema.performance_col]
        .rank(method='average', ascending=True)
    )

    known_metafeatures = set(SurrogateMetafeaturesSchema().to_list())
    feature_cols = [c for c in data.columns if c in known_metafeatures]

    base_cols = [
        schema.data_col, schema.rep_col, schema.n_random_warm_starts_col,
        schema.tuner_col, schema.label_col, BENCHMARK_ID_COL,
    ]
    base_cols = [c for c in base_cols if c in data.columns]
    result = data[base_cols + [c for c in feature_cols if c not in base_cols]].copy()

    if tuner_encoding_method == 'ordinal':
        result['tuner_encoded'] = LabelEncoder().fit_transform(result[schema.tuner_col])
        feature_cols = feature_cols + ['tuner_encoded']
    elif tuner_encoding_method == 'one_hot':
        dummies = pd.get_dummies(result[schema.tuner_col], prefix='tuner', drop_first=False)
        result = pd.concat([result, dummies], axis=1)
        feature_cols = feature_cols + list(dummies.columns)
    else:
        raise ValueError(
            f"Unknown tuner_encoding_method '{tuner_encoding_method}'. "
            "Must be 'ordinal' or 'one_hot'."
        )

    ws_col = schema.n_random_warm_starts_col
    if ws_col in result.columns and ws_col not in feature_cols:
        feature_cols.append(ws_col)

    result[schema.ranking_group_col] = (
        result[schema.data_col].astype(str) + '_'
        + result[schema.rep_col].astype(str) + '_'
        + result[ws_col].astype(str)
    )
    result['split_group'] = (
        result[schema.data_col].astype(str) + '_'
        + result[ws_col].astype(str)
    )

    return result, feature_cols


def split_data(
    data: pd.DataFrame,
    strategy: Literal['random', 'synthetic_train_real_test'],
    config: LTRConfig,
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

    Returns:
        ``(train_data, val_data, test_data)``

    Raises:
        ValueError: Strategy is ``'synthetic_train_real_test'`` but no
            synthetic rows exist.
    """
    val_prop = config.val_size / (config.train_size + config.val_size)

    if strategy == 'synthetic_train_real_test':
        is_synthetic = data[BENCHMARK_ID_COL] == SYNTHETIC_BENCHMARK
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
