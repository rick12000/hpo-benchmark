from pydantic import BaseModel
from typing import Any, Dict, List


class SearchSpaceMetafeaturesSchema(BaseModel):
    """Schema for search space metafeature column names.
    
    These features describe the hyperparameter search space structure and dimensionality.
    """
    
    n_integer_hyperparameters: str = "n_integer_hyperparameters"
    n_float_hyperparameters: str = "n_float_hyperparameters"
    n_categorical_hyperparameters: str = "n_categorical_hyperparameters"
    ratio_continuous_hyperparameters: str = "ratio_continuous_hyperparameters"
    ratio_categorical_hyperparameters: str = "ratio_categorical_hyperparameters"
    avg_categorical_cardinality: str = "avg_categorical_cardinality"
    min_categorical_cardinality: str = "min_categorical_cardinality"
    max_categorical_cardinality: str = "max_categorical_cardinality"
    total_search_space_combinations: str = "total_search_space_combinations"
    
    def to_list(self) -> List[str]:
        """Get all search space metafeature column names as a list."""
        return list(self.model_dump().values())


class DatasetMetafeaturesSchema(BaseModel):
    """Schema for dataset metafeature column names.
    
    These features describe the dataset characteristics (samples, features, classes, etc.).
    By default, this is a placeholder - actual dataset metafeatures are extracted
    from the objective function's get_metafeatures() method.
    """
    
    n_samples: str = "n_samples"
    n_features: str = "n_features"
    n_classes: str = "n_classes"
    
    def to_list(self) -> List[str]:
        """Get all dataset metafeature column names as a list."""
        return list(self.model_dump().values())


class BenchmarkDataSchema(BaseModel):
    """Schema defining column names for benchmark experiment data.

    Args:
        cumulative_coverage_error_col: Column for cumulative coverage error metrics.
        rolling_coverage_error_col: Column for rolling window coverage error.
        rep_col: Column for experiment repetition number.
        perf_col: Column for performance metric values.
        tuner_col: Column for tuner configuration identifier.
        bench_col: Column for benchmark suite name.
        data_col: Column for dataset identifier.
        sampler_col: Column for conformal sampler type.
        confidence_level_col: Column for confidence level values.
        estimator_architecture_col: Column for quantile estimator architecture.
        sampler_n_quantiles_col: Column for number of quantiles used.
        sampler_adapter_col: Column for adaptive conformal method.
        tuner_searcher_tuning_framework_col: Column for searcher tuning framework.
        n_pre_conformal_trials_col: Column for pre-conformal trial count.
        data_size_col: Column for dataset size.
        tuning_iterations_col: Column for tuning iteration count.
        estimator_error_col: Column for quantile estimator error.
        breach_col: Column for coverage breach status.
        runtime_unit: Base name for runtime columns.
        iter_unit: Base name for iteration columns.
        norm_runtime_unit: Name for normalized runtime columns.
        norm_iter_unit: Name for normalized iteration columns.
    """

    # Coverage error columns
    cumulative_coverage_error_col: str = "cumulative_coverage_error"
    rolling_coverage_error_col: str = "rolling_coverage_error"
    rep_col: str = "repetition"
    perf_col: str = "performance"
    tuner_col: str = "tuner"
    bench_col: str = "benchmark_identifier"
    data_col: str = "dataset"
    sampler_col: str = "sampler"
    confidence_level_col: str = "confidence_level"
    estimator_architecture_col: str = "estimator_architecture"
    sampler_n_quantiles_col: str = "sampler_n_quantiles"
    sampler_adapter_col: str = "sampler_adapter"
    tuner_searcher_tuning_framework_col: str = "tuner_searcher_tuning_framework"
    n_pre_conformal_trials_col: str = "n_pre_conformal_trials"
    data_size_col: str = "data_size"
    tuning_iterations_col: str = "tuning_iterations"
    estimator_error_col: str = "mean_pinball_loss"
    breach_col: str = "breach_status"
    n_random_warm_starts_col: str = "n_random_warm_starts"

    runtime_unit: str = "runtime"
    iter_unit: str = "iteration"
    norm_runtime_unit: str = f"normalized_{runtime_unit}"
    norm_iter_unit: str = f"normalized_{iter_unit}"
    performance_col: str = "performance"
    ranking_group_col: str = "ranking_group"
    label_col: str = "label"
    predicted_score_col: str = "predicted_score"
    
    # Nested schemas for specific use cases
    search_space_metafeatures: SearchSpaceMetafeaturesSchema = SearchSpaceMetafeaturesSchema()
    dataset_metafeatures: DatasetMetafeaturesSchema = DatasetMetafeaturesSchema()

    def to_list(self) -> List[str]:
        """Convert all schema field values to a list.

        Returns:
            List of all column names and units defined in the schema.
        """
        field_values: Dict[str, Any]
        field_values = self.model_dump()
        return list(field_values.values())
