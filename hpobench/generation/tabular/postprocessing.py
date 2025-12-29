import numpy as np
import pandas as pd
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class DatasetPostProcessor:
    def __init__(
        self,
        apply_quantization: bool,
        quantization_probability: float,
        apply_warping: bool,
        warping_probability: float,
        apply_missingness: bool,
        missingness_probability: float,
        apply_scaling: bool,
        seed: int,
    ):
        self.apply_quantization = apply_quantization
        self.quantization_probability = quantization_probability
        self.apply_warping = apply_warping
        self.warping_probability = warping_probability
        self.apply_missingness = apply_missingness
        self.missingness_probability = missingness_probability
        self.apply_scaling = apply_scaling
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def process(self, features: np.ndarray, targets: np.ndarray, task_type: str = "regression") -> Tuple[pd.DataFrame, pd.DataFrame]:
        features_processed = features.copy()
        targets_processed = targets.copy()
        
        if self.apply_quantization:
            features_processed = self._apply_quantization(features_processed)
            if task_type == "regression":
                targets_processed = self._apply_quantization(targets_processed)
        
        if self.apply_warping:
            features_processed = self._apply_warping(features_processed)
            if task_type == "regression":
                targets_processed = self._apply_warping(targets_processed)
        
        if self.apply_scaling:
            features_processed = self._apply_scaling(features_processed)
            if task_type == "regression":
                targets_processed = self._apply_scaling(targets_processed)
        
        if task_type == "classification":
            targets_processed = self._convert_to_classification(targets_processed)
        
        num_features = features_processed.shape[1]
        num_targets = targets_processed.shape[1]
        
        feature_names = [f"feature_{i}" for i in range(num_features)]
        target_names = [f"target_{i}" for i in range(num_targets)]
        
        features_df = pd.DataFrame(features_processed, columns=feature_names)
        targets_df = pd.DataFrame(targets_processed, columns=target_names)
        
        if self.apply_missingness:
            features_df = self._apply_missingness_to_df(features_df)
            targets_df = self._apply_missingness_to_df(targets_df)
        
        return features_df, targets_df
    
    def _apply_quantization(self, data: np.ndarray) -> np.ndarray:
        num_cols = data.shape[1]
        quantized_data = data.copy()
        
        for col_idx in range(num_cols):
            if self.rng.random() < self.quantization_probability:
                num_levels = self.rng.choice([2, 3, 5, 10, 20])
                col_min = data[:, col_idx].min()
                col_max = data[:, col_idx].max()
                
                if col_max > col_min:
                    normalized = (data[:, col_idx] - col_min) / (col_max - col_min)
                    quantized = np.floor(normalized * num_levels)
                    quantized = np.clip(quantized, 0, num_levels - 1)
                    quantized_data[:, col_idx] = quantized * (col_max - col_min) / num_levels + col_min
        
        return quantized_data
    
    def _apply_warping(self, data: np.ndarray) -> np.ndarray:
        num_cols = data.shape[1]
        warped_data = data.copy()
        
        warping_functions = [
            lambda x: np.sign(x) * np.power(np.abs(x), 2),
            lambda x: np.sign(x) * np.sqrt(np.abs(x) + 1e-8),
            lambda x: np.log(np.abs(x) + 1),
            lambda x: np.exp(np.clip(x, -10, 10)),
            lambda x: np.tanh(x),
        ]
        
        for col_idx in range(num_cols):
            if self.rng.random() < self.warping_probability:
                warp_func = self.rng.choice(warping_functions)
                try:
                    warped_col = warp_func(data[:, col_idx])
                    if not np.any(np.isnan(warped_col)) and not np.any(np.isinf(warped_col)):
                        warped_data[:, col_idx] = warped_col
                except Exception as e:
                    logger.warning(f"Warping failed for column {col_idx}: {e}")
        
        return warped_data
    
    def _apply_scaling(self, data: np.ndarray) -> np.ndarray:
        num_cols = data.shape[1]
        scaled_data = data.copy()
        
        for col_idx in range(num_cols):
            col_std = data[:, col_idx].std()
            if col_std > 1e-8:
                scale_factor = self.rng.uniform(0.1, 10.0)
                scaled_data[:, col_idx] = data[:, col_idx] * scale_factor
        
        return scaled_data
    
    def _apply_missingness_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df_with_missing = df.copy()
        
        for col in df.columns:
            if self.rng.random() < self.missingness_probability:
                missing_rate = self.rng.uniform(0.01, 0.2)
                num_missing = int(len(df) * missing_rate)
                missing_indices = self.rng.choice(len(df), size=num_missing, replace=False)
                df_with_missing.loc[missing_indices, col] = np.nan
        
        return df_with_missing
    
    def _convert_to_classification(self, targets: np.ndarray) -> np.ndarray:
        targets_class = targets.copy()
        
        for col_idx in range(targets.shape[1]):
            target_col = targets[:, col_idx]
            n_classes = self.rng.choice([2, 3, 4, 5])
            
            quantiles = np.linspace(0, 100, n_classes + 1)
            thresholds = np.percentile(target_col, quantiles)
            
            class_labels = np.digitize(target_col, thresholds[1:-1])
            targets_class[:, col_idx] = class_labels
        
        return targets_class


class FeatureTargetSelector:
    def __init__(self, num_features: int, num_targets: int, seed: int):
        self.num_features = num_features
        self.num_targets = num_targets
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def select_nodes(self, num_available_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
        if num_available_nodes < self.num_features + self.num_targets:
            raise ValueError(
                f"Not enough nodes ({num_available_nodes}) for "
                f"{self.num_features} features and {self.num_targets} targets"
            )
        
        all_nodes = np.arange(num_available_nodes)
        self.rng.shuffle(all_nodes)
        
        target_indices = all_nodes[:self.num_targets]
        feature_indices = all_nodes[self.num_targets:self.num_targets + self.num_features]
        
        return feature_indices, target_indices

