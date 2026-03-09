from pydantic import BaseModel
from typing import Any, Dict, List


class SurrogateMetafeaturesSchema(BaseModel):
    """Schema for surrogate data metafeature column names.
    
    These features describe the characteristics of the surrogate data (hyperparameter
    configurations and their associated performances), which directly represent the
    optimization landscape that tuners operate on.
    
    Surrogate metafeatures capture:
    - Size of the surrogate dataset (number of config-performance pairs, and dimensions after preprocessing)
    - Performance landscape statistics (mean, std, range, distribution shape)
    - Relationships between hyperparameters and performance
    - Conditional performance characteristics and heteroscedasticity
    - Mutual information between features and target
    """
    
    # Size metafeatures
    n_hyperparameters: str = "n_hyperparameters"
    total_rows: str = "total_rows"
    total_columns: str = "total_columns"
    
    # Hyperparameter type statistics
    n_integer_hyperparameters: str = "n_integer_hyperparameters"
    n_float_hyperparameters: str = "n_float_hyperparameters"
    n_binary_categorical_hyperparameters: str = "n_binary_categorical_hyperparameters"
    n_multicategory_hyperparameters: str = "n_multicategory_hyperparameters"
    
    # Performance statistics
    performance_mean: str = "performance_mean"
    performance_std: str = "performance_std"
    performance_min: str = "performance_min"
    performance_max: str = "performance_max"
    performance_range: str = "performance_range"
    performance_skewness: str = "performance_skewness"
    performance_kurtosis: str = "performance_kurtosis"
    
    # Conditional performance characteristics
    conditional_performance_skewness: str = "conditional_performance_skewness"
    performance_heteroscedasticity: str = "performance_heteroscedasticity"
    
    # Best performance (for reference)
    best_performance: str = "best_performance"
    
    # Correlation between hyperparameters and performance
    avg_config_performance_correlation: str = "avg_config_performance_correlation"
    max_config_performance_correlation: str = "max_config_performance_correlation"
    
    # Mutual information between features and target
    max_mi_with_target: str = "max_mi_with_target"
    min_mi_with_target: str = "min_mi_with_target"
    avg_mi_with_target: str = "avg_mi_with_target"
    
    # Mutual information between features
    max_mi_between_features: str = "max_mi_between_features"
    min_mi_between_features: str = "min_mi_between_features"
    avg_mi_between_features: str = "avg_mi_between_features"
    
    def to_list(self) -> List[str]:
        """Get all surrogate metafeature column names as a list."""
        return list(self.model_dump().values())



class BenchmarkDataSchema(BaseModel):
    """Schema defining column names for benchmark experiment data.
    
    Defines taxonomy of columns:
    - Input: Raw columns that must exist in raw data
    - Grouping: Columns that define experiment groups for ranking
    - Labels: Computed ranking labels
    - Engineered: Columns derived from input for pipeline operations
    - Metadata: Identification columns for filtering/tracking
    """

    # ===== INPUT COLUMNS (must exist in raw data) =====
    rep_col: str = "repetition"
    performance_col: str = "performance"
    tuner_col: str = "tuner"
    data_col: str = "dataset"
    n_random_warm_starts_col: str = "n_random_warm_starts"
    warm_start_strategy_col: str = "warm_start_strategy"
    
    # ===== LABEL COLUMNS (computed by preprocessing) =====
    label_col: str = "label"
    
    # ===== ENGINEERED COLUMNS (derived by preprocessing) =====
    ranking_group_col: str = "ranking_group"
    split_group_col: str = "split_group"
    
    # ===== METADATA COLUMNS (filtering/identification) =====
    benchmark_identifier_col: str = "benchmark_identifier"
    
    # ===== COLUMN LISTS (for pipeline operations) =====
    rank_group_cols: List[str] = []
    split_group_cols: List[str] = []
    
    def __init__(self, **data):
        super().__init__(**data)
        # Set column lists based on individual column names
        self.rank_group_cols = [
            self.data_col,
            self.warm_start_strategy_col,
            self.n_random_warm_starts_col,
            self.rep_col,
        ]
        self.split_group_cols = [
            self.data_col,
            self.warm_start_strategy_col,
            self.n_random_warm_starts_col,
        ]

    def to_list(self) -> List[str]:
        """Convert all schema field values to a list.

        Returns:
            List of all column names defined in the schema.
        """
        return list(self.model_dump().values())


class Aliases(BaseModel):
    """Human-readable aliases for various benchmark components.

    Args:
        sampler_aliases: Short names for conformal prediction samplers.
        architecture_aliases: Short names for quantile estimator architectures.
        benchmark_aliases: Display names for benchmark suites.
    """

    sampler_aliases: Dict[str, str] = {
        "ThompsonSampler": "TS",
        "ExpectedImprovementSampler": "EI",
        "LowerBoundSampler": "LBS",
        "PessimisticLowerBoundSampler": "PLBS",
    }
    architecture_aliases: Dict[str, str] = {
        "qknn": "QKNN",
        "qgp": "QGP",
        "ql": "QL",
        "qrf": "QRF",
        "qgbm": "QGBM",
        "qens5": "QE",
    }
    benchmark_aliases: Dict[str, str] = {
        "synthetic_tabular": "Synthetic-Tabular",
    }
