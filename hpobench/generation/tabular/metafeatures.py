import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging
from hpobench.config.schema import DatasetMetafeaturesSchema

logger = logging.getLogger(__name__)


def calculate_metafeatures(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    task_type: str,
    schema: Optional[DatasetMetafeaturesSchema] = None,
) -> Dict:
    """
    Calculate basic metafeatures for a dataset.
    
    This is a minimal implementation for analysis purposes only.
    Complex metafeatures (heteroscedasticity, GP-based, etc.) have been removed
    as they are not needed for dataset generation.
    
    Args:
        features: Feature DataFrame
        targets: Target DataFrame
        task_type: "regression" or "classification"
        schema: Optional DatasetMetafeaturesSchema for column naming
        
    Returns:
        Dictionary of basic metafeatures
    """
    if schema is None:
        schema = DatasetMetafeaturesSchema()
    
    X = features.values
    y = targets.values.ravel()
    
    n_samples = len(X)
    n_features = X.shape[1]
    
    metafeatures = {
        schema.n_samples: n_samples,
        schema.n_features: n_features,
        "task_type": task_type,
    }
    
    if task_type == "classification":
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        metafeatures[schema.n_classes] = n_classes
        
        if n_classes > 1:
            class_counts = np.bincount(y.astype(int))
            class_imbalance = np.std(class_counts) / (np.mean(class_counts) + 1e-10)
            metafeatures["class_imbalance"] = float(class_imbalance)
        else:
            metafeatures["class_imbalance"] = 0.0
    else:
        metafeatures[schema.n_classes] = 0
        metafeatures["class_imbalance"] = 0.0
        
        y_std = np.std(y)
        y_mean = np.mean(y)
        metafeatures["target_normalized_std"] = float(y_std / (abs(y_mean) + 1e-10))
    
    return metafeatures


# Backward compatibility alias
calculate_basic_metafeatures = calculate_metafeatures
