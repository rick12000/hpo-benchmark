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
    """Compute within-group ranking labels based on performance."""
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
    """Create concatenated identifier columns for ranking and splitting."""
    data[ranking_group_id_col] = data[rank_group_cols].astype(str).agg('_'.join, axis=1)
    data[split_group_id_col] = data[split_group_cols].astype(str).agg('_'.join, axis=1)
    return [ranking_group_id_col, split_group_id_col]


def _encode_tuner(
    data: pd.DataFrame,
    tuner_col: str,
    method: TunerEncoding,
) -> list[str]:
    """Encode tuners as numeric or one-hot features."""
    if method == 'ordinal':
        data['tuner_encoded'] = LabelEncoder().fit_transform(data[tuner_col])
        return ['tuner_encoded']
    elif method == 'one_hot':
        dummies = pd.get_dummies(data[tuner_col], prefix='tuner', drop_first=False)
        for col in dummies.columns:
            data[col] = dummies[col]
        return list(dummies.columns)
    else:
        raise ValueError(f'Unknown tuner encoding method {method!r}. Must be ordinal or one_hot.')


def prepare_data(
    raw_data: pd.DataFrame,
    schema: BenchmarkDataSchema,
    metafeatures_schema: SurrogateMetafeaturesSchema,
    tuner_encoding_method: TunerEncoding = 'ordinal',
) -> tuple[pd.DataFrame, list[str]]:
    """Prepare benchmark data for learning-to-rank training.
    
    Applies preprocessing: rank computation → engineered columns → tuner encoding.
    """
    if raw_data.empty:
        raise ValueError('No data available for preparation')

    data = raw_data.copy()

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
    partition: Partition,
    strategy: SplitStrategy,
    train_size: float,
    val_size: float,
    random_state: int,
    synthetic_benchmark_id: str,
    schema: BenchmarkDataSchema,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train/validation/test sets.

    Handles different partitioning strategies while ensuring a consistent test set for real-world data evaluation.

    Strategies for 'all' partition:
    - 'random': 
        Train = Synthetic Train + Real Train
        Val = Synthetic Val + Real Val
        Test = Synthetic Test + Real Test
    - 'synthetic_train_real_test': 
        Train = All Synthetic (Train+Val+Test) + Real Train
        Val = Real Val
        Test = Real Test

    For 'synthetic' or 'real' partitions, performs a standard Train/Val/Test split.
    The Real data split is consistent across all strategies to ensure comparable evaluation metrics.
    """
    is_synthetic = data[schema.benchmark_identifier_col] == synthetic_benchmark_id
    synthetic = data[is_synthetic]
    real = data[~is_synthetic]

    real_train = pd.DataFrame()
    real_val = pd.DataFrame()
    real_test = pd.DataFrame()

    if not real.empty:
        test_prop = 1.0 - (train_size + val_size)
        if test_prop <= 0:
            raise ValueError(f"train_size ({train_size}) + val_size ({val_size}) must be < 1.0")

        # Split Real -> Dev (Train+Val) + Test
        test_splitter = GroupShuffleSplit(n_splits=1, test_size=test_prop, random_state=random_state)
        dev_idx, test_idx = next(test_splitter.split(real, groups=real[schema.split_group_id_col]))
        real_dev = real.iloc[dev_idx]
        real_test = real.iloc[test_idx]

        # Split Dev -> Train + Val
        val_prop_dev = val_size / (train_size + val_size)
        dev_splitter = GroupShuffleSplit(n_splits=1, test_size=val_prop_dev, random_state=random_state)
        train_idx, val_idx = next(dev_splitter.split(real_dev, groups=real_dev[schema.split_group_id_col]))
        real_train = real_dev.iloc[train_idx]
        real_val = real_dev.iloc[val_idx]

    syn_train = pd.DataFrame()
    syn_val = pd.DataFrame()
    syn_test = pd.DataFrame()

    if not synthetic.empty:
        # Split synthetic data only if needed for 'synthetic' partition or 'random' strategy
        if partition == 'synthetic' or (partition == 'all' and strategy == 'random'):
            test_prop = 1.0 - (train_size + val_size)
            if test_prop <= 0:
                raise ValueError(f"train_size ({train_size}) + val_size ({val_size}) must be < 1.0")

            splitter = GroupShuffleSplit(n_splits=1, test_size=test_prop, random_state=random_state)
            dev_idx, test_idx = next(splitter.split(synthetic, groups=synthetic[schema.split_group_id_col]))
            syn_dev = synthetic.iloc[dev_idx]
            syn_test = synthetic.iloc[test_idx]

            val_prop_dev = val_size / (train_size + val_size)
            dev_splitter = GroupShuffleSplit(n_splits=1, test_size=val_prop_dev, random_state=random_state)
            train_idx, val_idx = next(dev_splitter.split(syn_dev, groups=syn_dev[schema.split_group_id_col]))
            syn_train = syn_dev.iloc[train_idx]
            syn_val = syn_dev.iloc[val_idx]

    if partition == 'synthetic':
        if synthetic.empty:
            raise ValueError("Partition is 'synthetic' but no synthetic data found.")
        return syn_train, syn_val, syn_test

    elif partition == 'real':
        if real.empty:
            raise ValueError("Partition is 'real' but no real data found.")
        return real_train, real_val, real_test

    elif partition == 'all':
        if strategy == 'synthetic_train_real_test':
            if synthetic.empty or real.empty:
                 raise ValueError("Strategy 'synthetic_train_real_test' requires both synthetic and real data.")
            
            return pd.concat([synthetic, real_train]), real_val, real_test
        
        elif strategy == 'random':
            if synthetic.empty or real.empty:
                 raise ValueError("Strategy 'random' with partition 'all' requires both synthetic and real data.")

            return (
                pd.concat([syn_train, real_train]),
                pd.concat([syn_val, real_val]),
                pd.concat([syn_test, real_test])
            )
        else:
            raise ValueError(f"Unknown strategy {strategy!r}")

    else:
        raise ValueError(f"Unknown partition {partition!r}")
