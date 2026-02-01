import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Union
import logging
from scipy import stats
from hpobench.config.schema import SurrogateMetafeaturesSchema

logger = logging.getLogger(__name__)


def calculate_surrogate_metafeatures(
    configs: List[Dict[str, Union[int, float, str]]],
    performances: List[float],
    schema: Optional[SurrogateMetafeaturesSchema] = None,
) -> Dict:
    """
    Calculate metafeatures for surrogate data (hyperparameter configs + performances).
    
    Treats the surrogate data as a tabular dataset where:
    - Rows = hyperparameter configurations
    - Columns = hyperparameter values + performance
    
    Args:
        configs: List of hyperparameter configuration dictionaries
        performances: List of performance values corresponding to each config
        schema: Optional SurrogateMetafeaturesSchema for column naming
        
    Returns:
        Dictionary of surrogate metafeatures
    """
    if schema is None:
        schema = SurrogateMetafeaturesSchema()
    
    if len(configs) == 0 or len(performances) == 0:
        logger.warning("Empty surrogate data provided, returning NaN metafeatures")
        return _get_nan_surrogate_metafeatures(schema)
    
    if len(configs) != len(performances):
        raise ValueError(
            f"Mismatch between configs ({len(configs)}) and performances ({len(performances)})"
        )
    
    # Convert to numpy arrays for easier computation
    performances_arr = np.array(performances)
    
    # Size metafeatures
    n_surrogate_samples = len(configs)
    n_hyperparameters = len(configs[0]) if configs else 0
    
    # Performance statistics
    performance_mean = float(np.mean(performances_arr))
    performance_std = float(np.std(performances_arr))
    performance_min = float(np.min(performances_arr))
    performance_max = float(np.max(performances_arr))
    performance_range = performance_max - performance_min
    
    # Distribution shape
    try:
        performance_skewness = float(stats.skew(performances_arr))
        performance_kurtosis = float(stats.kurtosis(performances_arr))
    except Exception as e:
        logger.warning(f"Failed to calculate skewness/kurtosis: {e}")
        performance_skewness = 0.0
        performance_kurtosis = 0.0
    
    # Best performance (minimum for minimization problems)
    best_performance = performance_min
    
    # Calculate correlation between hyperparameters and performance
    correlations = []
    
    # Convert configs to DataFrame for easier correlation calculation
    try:
        config_df = pd.DataFrame(configs)
        
        # Only calculate correlations for numeric columns
        for col in config_df.columns:
            try:
                # Try to convert to numeric
                numeric_col = pd.to_numeric(config_df[col], errors='coerce')
                
                # Skip if all NaN after conversion
                if numeric_col.notna().sum() > 1:
                    # Calculate correlation with performance
                    corr = numeric_col.corr(pd.Series(performances_arr))
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
            except Exception:
                # Skip non-numeric or problematic columns
                continue
        
        if len(correlations) > 0:
            avg_config_performance_correlation = float(np.mean(correlations))
            max_config_performance_correlation = float(np.max(correlations))
        else:
            avg_config_performance_correlation = 0.0
            max_config_performance_correlation = 0.0
    except Exception as e:
        logger.warning(f"Failed to calculate config-performance correlations: {e}")
        avg_config_performance_correlation = 0.0
        max_config_performance_correlation = 0.0
    
    metafeatures = {
        schema.n_surrogate_samples: n_surrogate_samples,
        schema.n_hyperparameters: n_hyperparameters,
        schema.performance_mean: performance_mean,
        schema.performance_std: performance_std,
        schema.performance_min: performance_min,
        schema.performance_max: performance_max,
        schema.performance_range: performance_range,
        schema.performance_skewness: performance_skewness,
        schema.performance_kurtosis: performance_kurtosis,
        schema.best_performance: best_performance,
        schema.avg_config_performance_correlation: avg_config_performance_correlation,
        schema.max_config_performance_correlation: max_config_performance_correlation,
    }
    
    return metafeatures


def _get_nan_surrogate_metafeatures(
    schema: Optional[SurrogateMetafeaturesSchema] = None,
) -> Dict:
    """Return a dictionary of surrogate metafeatures filled with NaN values.
    
    Args:
        schema: Optional SurrogateMetafeaturesSchema for column naming
        
    Returns:
        Dictionary with surrogate metafeature keys set to NaN.
    """
    if schema is None:
        schema = SurrogateMetafeaturesSchema()
    
    return {
        schema.n_surrogate_samples: np.nan,
        schema.n_hyperparameters: np.nan,
        schema.performance_mean: np.nan,
        schema.performance_std: np.nan,
        schema.performance_min: np.nan,
        schema.performance_max: np.nan,
        schema.performance_range: np.nan,
        schema.performance_skewness: np.nan,
        schema.performance_kurtosis: np.nan,
        schema.best_performance: np.nan,
        schema.avg_config_performance_correlation: np.nan,
        schema.max_config_performance_correlation: np.nan,
    }
