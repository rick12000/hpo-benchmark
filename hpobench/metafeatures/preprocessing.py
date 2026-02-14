"""Preprocessing utilities for metafeature calculation.

This module provides preprocessing functions to prepare hyperparameter configurations
for metafeature calculation, including one-hot encoding and normalization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import logging

from hpobench.config.types import IntRange, FloatRange, CategoricalRange

logger = logging.getLogger(__name__)


def preprocess_for_metafeatures(
    configs: List[Dict[str, Union[int, float, str]]],
    search_space: Optional[Dict[str, Union[IntRange, FloatRange, CategoricalRange]]] = None,
) -> Tuple[np.ndarray, Dict]:
    """Preprocess hyperparameter configurations for metafeature calculation.
    
    Applies the following transformations:
    1. One-hot encode categorical features
    2. Identify binary features (including one-hot encoded)
    3. Normalize all non-binary features using StandardScaler
    
    Args:
        configs: List of hyperparameter configuration dictionaries
        search_space: Optional search space to identify categorical features
        
    Returns:
        Tuple of (preprocessed_array, metadata_dict) where metadata contains:
            - original_feature_names: Original feature names
            - preprocessed_feature_names: Names after one-hot encoding
            - categorical_features: List of categorical feature names
            - binary_features: List of binary feature names (after encoding)
            - normalized_features: List of normalized feature names
            - scaler: Fitted StandardScaler (if normalization applied)
            - encoder: Fitted OneHotEncoder (if encoding applied)
    """
    if len(configs) == 0:
        raise ValueError("Cannot preprocess empty configuration list")
    
    # Convert to DataFrame
    configs_df = pd.DataFrame(configs)
    original_feature_names = configs_df.columns.tolist()
    
    # Identify categorical features
    categorical_features = []
    if search_space is not None:
        # Use search space to identify categoricals
        for hp_name, hp_range in search_space.items():
            if isinstance(hp_range, CategoricalRange):
                if hp_name in configs_df.columns:
                    categorical_features.append(hp_name)
    else:
        # Infer categoricals from data
        for col in configs_df.columns:
            # Check if column is non-numeric or has few unique values
            try:
                pd.to_numeric(configs_df[col])
            except (ValueError, TypeError):
                categorical_features.append(col)
                continue
            
            # If numeric but has few unique values, might be categorical
            if configs_df[col].nunique() <= 10:
                categorical_features.append(col)
    
    
    # Separate categorical and numeric features
    numeric_features = [col for col in configs_df.columns if col not in categorical_features]
    
    # Initialize metadata
    metadata = {
        'original_feature_names': original_feature_names,
        'categorical_features': categorical_features,
        'numeric_features': numeric_features,
    }
    
    # Step 1: One-hot encode categorical features
    if len(categorical_features) > 0:
        # Prepare categorical data
        cat_data = configs_df[categorical_features].copy()
        
        # Convert to string to handle mixed types
        for col in categorical_features:
            cat_data[col] = cat_data[col].astype(str)
        
        # One-hot encode
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_cat = encoder.fit_transform(cat_data)
        
        # Get feature names after encoding
        encoded_feature_names = []
        for i, col in enumerate(categorical_features):
            categories = encoder.categories_[i]
            for cat in categories:
                encoded_feature_names.append(f"{col}_{cat}")
        
        # Create DataFrame with encoded features
        encoded_df = pd.DataFrame(
            encoded_cat,
            columns=encoded_feature_names,
            index=configs_df.index
        )
        
        metadata['encoder'] = encoder
        metadata['encoded_feature_names'] = encoded_feature_names
    else:
        encoded_df = pd.DataFrame(index=configs_df.index)
        metadata['encoder'] = None
        metadata['encoded_feature_names'] = []
    
    # Step 2: Combine numeric and encoded features
    if len(numeric_features) > 0:
        numeric_df = configs_df[numeric_features].copy()
        
        # Convert to numeric, handling any errors
        for col in numeric_features:
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
        
        # Fill any NaN values with 0
        numeric_df = numeric_df.fillna(0)
        
        combined_df = pd.concat([numeric_df, encoded_df], axis=1)
    else:
        combined_df = encoded_df
    
    # Step 3: Identify binary features (don't normalize these)
    binary_features = []
    for col in combined_df.columns:
        unique_vals = combined_df[col].dropna().unique()
        if len(unique_vals) <= 2:
            # Check if values are 0/1 or similar binary
            if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                binary_features.append(col)
    
    logger.debug(f"Identified {len(binary_features)} binary features")
    metadata['binary_features'] = binary_features
    
    # Step 4: Normalize non-binary features
    features_to_normalize = [col for col in combined_df.columns if col not in binary_features]
    
    if len(features_to_normalize) > 0:
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(combined_df[features_to_normalize])
        
        # Replace normalized features
        for i, col in enumerate(features_to_normalize):
            combined_df[col] = normalized_data[:, i]
        
        metadata['scaler'] = scaler
        metadata['normalized_features'] = features_to_normalize
    else:
        metadata['scaler'] = None
        metadata['normalized_features'] = []
    
    # Final feature names
    metadata['preprocessed_feature_names'] = combined_df.columns.tolist()
    
    # Convert to numpy array
    preprocessed_array = combined_df.values
    
    logger.debug(
        f"Preprocessing complete: {len(original_feature_names)} original features → "
        f"{preprocessed_array.shape[1]} preprocessed features "
        f"({len(categorical_features)} categorical, {len(binary_features)} binary, "
        f"{len(features_to_normalize)} normalized)"
    )
    
    return preprocessed_array, metadata


def preprocess_configs_simple(
    configs: List[Dict[str, Union[int, float, str]]],
) -> np.ndarray:
    """Simple preprocessing that converts configs to numeric array.
    
    This is a lightweight version that just converts to numeric without
    one-hot encoding or normalization. Useful for quick conversions.
    
    Args:
        configs: List of hyperparameter configuration dictionaries
        
    Returns:
        Numeric array of shape (n_configs, n_features)
    """
    if len(configs) == 0:
        raise ValueError("Cannot preprocess empty configuration list")
    
    configs_df = pd.DataFrame(configs)
    
    # Convert all columns to numeric, using label encoding for non-numeric
    numeric_df = configs_df.copy()
    for col in numeric_df.columns:
        try:
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
        except:
            # Use label encoding for non-numeric
            numeric_df[col] = pd.factorize(numeric_df[col])[0]
    
    # Fill NaN with 0
    numeric_df = numeric_df.fillna(0)
    
    return numeric_df.values


def identify_categorical_from_search_space(
    search_space: Dict[str, Union[IntRange, FloatRange, CategoricalRange]]
) -> List[str]:
    """Identify categorical hyperparameters from search space.
    
    Args:
        search_space: Search space dictionary
        
    Returns:
        List of categorical hyperparameter names
    """
    categorical_features = []
    for hp_name, hp_range in search_space.items():
        if isinstance(hp_range, CategoricalRange):
            categorical_features.append(hp_name)
    return categorical_features
