import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class DatasetPostProcessor:
    def __init__(
        self,
        apply_standardization: bool,
        classification_method: str,
        seed: int,
    ):
        self.apply_standardization = apply_standardization
        self.classification_method = classification_method
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def process(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        task_type: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply minimal post-processing to features and targets.
        
        Args:
            features: Shape (num_samples, num_features)
            targets: Shape (num_samples, num_targets)
            task_type: "regression" or "classification"
            
        Returns:
            features_df: DataFrame with feature columns
            targets_df: DataFrame with target columns
        """
        features_processed = features.copy()
        targets_processed = targets.copy()
        
        if self.apply_standardization:
            features_processed = self._standardize(features_processed)
            if task_type == "regression":
                targets_processed = self._standardize(targets_processed)
        
        if task_type == "classification":
            targets_processed = self._convert_to_classification(targets_processed)
        
        num_features = features_processed.shape[1]
        num_targets = targets_processed.shape[1]
        
        feature_names = [f"feature_{i}" for i in range(num_features)]
        target_names = [f"target_{i}" for i in range(num_targets)]
        
        features_df = pd.DataFrame(features_processed, columns=feature_names)
        targets_df = pd.DataFrame(targets_processed, columns=target_names)
        
        return features_df, targets_df
    
    def _standardize(self, data: np.ndarray) -> np.ndarray:
        """Standardize data to zero mean and unit variance."""
        scaler = StandardScaler()
        return scaler.fit_transform(data)
    
    def _convert_to_classification(self, targets: np.ndarray) -> np.ndarray:
        """Convert continuous targets to classification labels."""
        targets_class = targets.copy()
        
        for col_idx in range(targets.shape[1]):
            target_col = targets[:, col_idx]
            
            num_unique = len(np.unique(target_col))
            if num_unique < 10:
                num_classes = min(num_unique, self.rng.choice([2, 3, 4]))
            else:
                num_classes = self.rng.choice([2, 3, 4, 5])
            
            if self.classification_method == "quantile":
                quantiles = np.linspace(0, 100, num_classes + 1)
                thresholds = np.percentile(target_col, quantiles)
                class_labels = np.digitize(target_col, thresholds[1:-1])
            elif self.classification_method == "kmeans":
                from sklearn.cluster import KMeans
                
                try:
                    kmeans = KMeans(n_clusters=num_classes, random_state=self.seed, n_init=10)
                    class_labels = kmeans.fit_predict(target_col.reshape(-1, 1))
                    
                    centers = kmeans.cluster_centers_.ravel()
                    label_mapping = np.argsort(np.argsort(centers))
                    class_labels = label_mapping[class_labels]
                except Exception as e:
                    logger.warning(f"K-means clustering failed: {e}. Using quantile binning.")
                    quantiles = np.linspace(0, 100, num_classes + 1)
                    thresholds = np.percentile(target_col, quantiles)
                    class_labels = np.digitize(target_col, thresholds[1:-1])
            else:
                raise ValueError(f"Unknown classification method: {self.classification_method}")
            
            targets_class[:, col_idx] = class_labels
        
        return targets_class
