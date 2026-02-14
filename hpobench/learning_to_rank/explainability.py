"""
Feature importance and explainability for learning-to-rank models.

This module provides tools for understanding and explaining learning-to-rank (LTR)
models used for HPO algorithm selection. It includes:

1. **ShaRP-based SHAP values**: Rank-based feature attributions using the ShaRP
   framework, which measures how features affect ranking positions rather than
   raw model scores.

2. **Rank-based Partial Dependence Plots (PDPs)**: Visualizations showing how
   individual features affect the ranking position of each tuner (algorithm)
   within ranking groups.

## Rank-Based Partial Dependence

Traditional partial dependence plots measure how features affect model predictions
(scores). In ranking contexts, we care about how features affect relative positions,
not absolute scores. Rank-based PDPs address this by:

### Mathematical Framework

For a tuner t and feature j, the rank-based PDP is:

    RankPDP_j^t(x_j) = (1/|G|) * Σ_{g ∈ G} rank_t(x_j, g)

where:
- rank_t(x_j, g) = rank of tuner t in group g when feature j = x_j
- G = set of all ranking groups where tuner t appears
- Ranks are computed within each group independently

### Key Properties

1. **Per-tuner analysis**: Each tuner gets its own PDP curves, revealing
   algorithm-specific feature sensitivities.

2. **Ranking-aware**: Measures rank impact, not score impact. A feature that
   changes scores by 10% might not change ranks at all if all competitors
   are affected equally.

3. **Group-structured**: Respects the LTR problem structure where items are
   ranked within groups (e.g., tuners ranked per dataset/configuration).

4. **Interpretable**: "When feature X increases by 1 unit, tuner A's average
   rank improves from 3rd to 2nd" is more actionable than score-based metrics.

### Usage Example

```python
from hpobench.learning_to_rank.explainability import (
    run_partial_dependence_analysis
)

# After running LTR analysis
models = {name: result.model for name, result in ltr_results.items()}
test_data = {name: result.test_data for name, result in ltr_results.items()}

# Compute and plot PDPs
pdp_results = run_partial_dependence_analysis(
    models_by_partition=models,
    test_data_by_partition=test_data,
    feature_cols=feature_cols,
    output_dir=Path('output/pdp'),
)
```

### Visualization

Each tuner gets an n×3 matrix of plots where:
- n = number of features
- 3 = number of data partitions (e.g., all_random, synthetic, real)
- Each subplot shows how one feature affects that tuner's rank
- Y-axis: Average rank (lower = better, axis inverted for clarity)
- X-axis: Feature values (sensible range based on data quantiles)

### Design Principles

The implementation follows SOLID principles:

- **Single Responsibility**: Separate classes for grid generation, rank computation,
  PDP computation, and plotting.
- **Open/Closed**: Extensible for different grid strategies and rank metrics.
- **Dependency Inversion**: Core logic depends on abstractions (dataclasses),
  not concrete implementations.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from dataclasses import dataclass
from typing import Literal
import logging

logger = logging.getLogger(__name__)


@dataclass
class SharpResults:
    """Results from ShaRP explainability analysis."""
    shap_values: np.ndarray
    feature_names: list[str]
    feature_matrix: np.ndarray
    base_value: float


@dataclass
class PartialDependenceResult:
    """Results from partial dependence analysis for a single feature and tuner.
    
    Attributes:
        feature_name: Name of the feature analyzed
        tuner_name: Name of the tuner (algorithm) analyzed
        x_values: Feature values at which PDP was computed
        rank_values: Average rank at each feature value
        rank_std: Standard deviation of ranks at each feature value
        n_groups: Number of ranking groups included in the analysis
    """
    feature_name: str
    tuner_name: str
    x_values: np.ndarray
    rank_values: np.ndarray
    rank_std: np.ndarray
    n_groups: int


@dataclass
class PartialDependenceResults:
    """Complete partial dependence analysis results for all features and tuners.
    
    Attributes:
        results: Dictionary mapping (feature_name, tuner_name) to PartialDependenceResult
        feature_names: List of all feature names analyzed
        tuner_names: List of all tuner names analyzed
        partition_name: Name of the data partition analyzed
    """
    results: dict[tuple[str, str], PartialDependenceResult]
    feature_names: list[str]
    tuner_names: list[str]
    partition_name: str


def get_feature_importance(model: xgb.Booster, importance_type: str = 'gain') -> pd.DataFrame:
    """Get feature importance from XGBoost model.
    
    Args:
        model: Trained XGBoost booster
        importance_type: 'gain', 'weight', or 'cover'
        
    Returns:
        DataFrame with feature importance sorted by importance
    """
    scores = model.get_score(importance_type=importance_type)
    df = pd.DataFrame([
        {'feature': k, 'importance': v}
        for k, v in scores.items()
    ])
    return df.sort_values('importance', ascending=False).reset_index(drop=True)


def compute_shap_values(
    model: xgb.Booster,
    data: pd.DataFrame,
    feature_cols: list[str],
    sample_size: int | None = None,
    random_state: int = 42,
) -> SharpResults:
    """Compute SHAP values using the ShaRP library.
    
    Args:
        model: Trained XGBoost booster
        data: Data to explain
        feature_cols: Feature column names
        sample_size: Sample size for perturbation (None uses default)
        random_state: Random seed
        
    Returns:
        SharpResults with SHAP values and metadata
    """
    try:
        from sharp import ShaRP
    except ImportError:
        raise ImportError("ShaRP required: pip install xai-sharp")
    
    X = data[feature_cols].values
    
    def score_fn(x: np.ndarray) -> np.ndarray:
        return model.predict(xgb.DMatrix(x))
    
    explainer = ShaRP(
        qoi='rank',
        target_function=score_fn,
        measure='shapley',
        sample_size=sample_size,
        replace=False,
        random_state=random_state,
        n_jobs=1,
        verbose=0,
    )
    explainer.fit(X, feature_names=feature_cols)
    shap_values = explainer.all(X=X)
    
    scores = score_fn(X)
    ranks = np.argsort(np.argsort(-scores)) + 1
    
    return SharpResults(
        shap_values=shap_values,
        feature_names=feature_cols,
        feature_matrix=X,
        base_value=ranks.mean(),
    )


def create_importance_summary(results: SharpResults) -> pd.DataFrame:
    """Create feature importance summary from SHAP values.
    
    Args:
        results: SharpResults from compute_shap_values
        
    Returns:
        DataFrame with importance statistics per feature
    """
    return pd.DataFrame({
        'feature': results.feature_names,
        'mean_abs_shap': np.abs(results.shap_values).mean(axis=0),
        'mean_shap': results.shap_values.mean(axis=0),
        'std_shap': results.shap_values.std(axis=0),
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)


def plot_global_importance(
    results: SharpResults,
    output_path: Path | None = None,
    top_k: int = 20,
) -> None:
    """Create bar plot of global feature importance.
    
    Args:
        results: SharpResults from compute_shap_values
        output_path: Path to save plot (displays if None)
        top_k: Number of top features to show
    """
    import matplotlib.pyplot as plt
    
    importance = np.abs(results.shap_values).mean(axis=0)
    indices = np.argsort(importance)[-top_k:][::-1]
    
    fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.3)))
    ax.barh(range(len(indices)), importance[indices], color='#ff0051')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([results.feature_names[i] for i in indices])
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title(f'Feature Importance (Top {top_k})')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_beeswarm(
    results: SharpResults,
    output_path: Path | None = None,
    top_k: int = 20,
) -> None:
    """Create beeswarm plot showing SHAP value distributions.
    
    Args:
        results: SharpResults from compute_shap_values
        output_path: Path to save plot (displays if None)
        top_k: Number of top features to show
    """
    import matplotlib.pyplot as plt
    
    importance = np.abs(results.shap_values).mean(axis=0)
    top_indices = np.argsort(importance)[-top_k:][::-1]
    
    fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.4)))
    rng = np.random.RandomState(42)
    
    for i, feat_idx in enumerate(top_indices):
        shap_vals = results.shap_values[:, feat_idx]
        feat_vals = results.feature_matrix[:, feat_idx]
        
        # Normalize feature values for coloring
        vmin, vmax = feat_vals.min(), feat_vals.max()
        if vmax > vmin:
            colors = (feat_vals - vmin) / (vmax - vmin)
        else:
            colors = np.zeros_like(feat_vals)
        
        y_pos = i + rng.uniform(-0.3, 0.3, len(shap_vals))
        ax.scatter(shap_vals, y_pos, c=colors, cmap='coolwarm', s=20, alpha=0.6)
    
    ax.set_yticks(range(len(top_indices)))
    ax.set_yticklabels([results.feature_names[i] for i in top_indices])
    ax.set_xlabel('SHAP value')
    ax.set_title(f'Feature Impact Distribution (Top {top_k})')
    ax.axvline(0, color='black', linewidth=0.8, alpha=0.5)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def run_explainability_analysis(
    model: xgb.Booster,
    test_data: pd.DataFrame,
    feature_cols: list[str],
    output_dir: Path | None = None,
    top_k: int = 20,
    sample_size: int | None = None,
) -> dict:
    """Run complete explainability analysis.
    
    Args:
        model: Trained XGBoost booster
        test_data: Test data with features
        feature_cols: Feature column names
        output_dir: Directory to save outputs
        top_k: Number of features to show in plots
        sample_size: Sample size for SHAP computation
        
    Returns:
        Dictionary with results and summary DataFrame
    """
    results = compute_shap_values(model, test_data, feature_cols, sample_size)
    summary = create_importance_summary(results)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        summary.to_csv(output_dir / 'feature_importance.csv', index=False)
        plot_global_importance(results, output_dir / 'importance_bar.png', top_k)
        plot_beeswarm(results, output_dir / 'importance_beeswarm.png', top_k)
    
    return {'shap_results': results, 'summary': summary}


# Backwards compatibility aliases
def create_xgboost_score_function(model: xgb.Booster, feature_cols: list[str]):
    """Create scoring function wrapper (backwards compatible)."""
    def score_function(X: np.ndarray) -> np.ndarray:
        return model.predict(xgb.DMatrix(X))
    return score_function


def compute_sharp_explanations(model, test_data, feature_cols, **kwargs):
    """Compute ShaRP explanations (backwards compatible)."""
    results = compute_shap_values(model, test_data, feature_cols, 
                                   kwargs.get('sample_size'), kwargs.get('random_state', 42))
    return {
        'sharp_values': results.shap_values,
        'feature_names': results.feature_names,
        'feature_matrix': results.feature_matrix,
        'base_value': results.base_value,
    }


def create_global_importance_plot(sharp_values, feature_names, output_path=None, top_k=20):
    """Create global importance plot (backwards compatible)."""
    results = SharpResults(sharp_values, feature_names, np.zeros((len(sharp_values), len(feature_names))), 0)
    plot_global_importance(results, output_path, top_k)


def create_beeswarm_plot(sharp_values, feature_matrix, feature_names, output_path=None, top_k=20):
    """Create beeswarm plot (backwards compatible)."""
    results = SharpResults(sharp_values, feature_names, feature_matrix, 0)
    plot_beeswarm(results, output_path, top_k)


def create_feature_importance_summary(sharp_values, feature_names, output_path=None):
    """Create feature importance summary (backwards compatible)."""
    results = SharpResults(sharp_values, feature_names, np.zeros((len(sharp_values), len(feature_names))), 0)
    summary = create_importance_summary(results)
    if output_path:
        summary.to_csv(output_path, index=False)
    return summary


def run_sharp_analysis(model, test_data, feature_cols, output_dir=None, **kwargs):
    """Run ShaRP analysis (backwards compatible)."""
    return run_explainability_analysis(
        model, test_data, feature_cols, output_dir,
        kwargs.get('top_k_features', 20), kwargs.get('sample_size')
    )


# ============================================================================
# Rank-Based Partial Dependence Analysis
# ============================================================================


class FeatureGridGenerator:
    """Generates appropriate value grids for features in partial dependence analysis.
    
    Follows Single Responsibility Principle: only responsible for grid generation.
    """
    
    def __init__(
        self,
        n_grid_points: int = 20,
        quantile_range: tuple[float, float] = (0.05, 0.95),
        categorical_threshold: int = 10,
    ):
        """Initialize grid generator.
        
        Args:
            n_grid_points: Number of grid points for continuous features
            quantile_range: (min, max) quantiles for continuous feature range
            categorical_threshold: Max unique values to treat as categorical
        """
        self.n_grid_points = n_grid_points
        self.quantile_range = quantile_range
        self.categorical_threshold = categorical_threshold
    
    def generate_grid(self, feature_values: pd.Series) -> np.ndarray:
        """Generate appropriate grid for a feature.
        
        Args:
            feature_values: Series of feature values from data
            
        Returns:
            Array of grid points for partial dependence computation
        """
        unique_values = feature_values.dropna().unique()
        n_unique = len(unique_values)
        
        # Categorical or low-cardinality features: use all unique values
        if n_unique <= self.categorical_threshold or feature_values.dtype == 'object':
            return np.sort(unique_values)
        
        # Continuous features: use quantile-based grid
        q_min, q_max = self.quantile_range
        x_min = feature_values.quantile(q_min)
        x_max = feature_values.quantile(q_max)
        
        # Handle edge case where all values are the same
        if x_min == x_max:
            return np.array([x_min])
        
        return np.linspace(x_min, x_max, self.n_grid_points)


class RankComputer:
    """Computes ranks within ranking groups for LTR models.
    
    Follows Single Responsibility Principle: only responsible for rank computation.
    """
    
    def __init__(self, ranking_group_col: str, tuner_col: str):
        """Initialize rank computer.
        
        Args:
            ranking_group_col: Column name for ranking groups
            tuner_col: Column name for tuner identifiers
        """
        self.ranking_group_col = ranking_group_col
        self.tuner_col = tuner_col
    
    def compute_ranks_by_group(
        self,
        data: pd.DataFrame,
        scores: np.ndarray,
    ) -> dict[str, dict[str, list[float]]]:
        """Compute ranks within each ranking group, organized by tuner.
        
        Args:
            data: DataFrame with ranking groups and tuner identifiers
            scores: Predicted scores for each row
            
        Returns:
            Dictionary mapping tuner_name -> {group_id: [ranks]}
        """
        data_with_scores = data.copy()
        data_with_scores['_pred_score'] = scores
        
        tuner_ranks = {}
        
        for group_id, group_df in data_with_scores.groupby(self.ranking_group_col):
            group_scores = group_df['_pred_score'].values
            # Rank: lower score gets higher rank (1 is best)
            # argsort(-scores) gives indices that would sort in descending order
            group_ranks = np.argsort(np.argsort(-group_scores)) + 1
            
            # Map ranks to tuners
            for idx, (_, row) in enumerate(group_df.iterrows()):
                tuner = row[self.tuner_col]
                if tuner not in tuner_ranks:
                    tuner_ranks[tuner] = {}
                if group_id not in tuner_ranks[tuner]:
                    tuner_ranks[tuner][group_id] = []
                tuner_ranks[tuner][group_id].append(float(group_ranks[idx]))
        
        return tuner_ranks


class PartialDependenceComputer:
    """Computes rank-based partial dependence for LTR models.
    
    Follows Open/Closed Principle: extensible for different PDP strategies.
    """
    
    def __init__(
        self,
        model: xgb.Booster,
        feature_cols: list[str],
        ranking_group_col: str,
        tuner_col: str,
        grid_generator: FeatureGridGenerator | None = None,
    ):
        """Initialize partial dependence computer.
        
        Args:
            model: Trained XGBoost booster
            feature_cols: List of feature column names
            ranking_group_col: Column name for ranking groups
            tuner_col: Column name for tuner identifiers
            grid_generator: Optional custom grid generator
        """
        self.model = model
        self.feature_cols = feature_cols
        self.ranking_group_col = ranking_group_col
        self.tuner_col = tuner_col
        self.grid_generator = grid_generator or FeatureGridGenerator()
        self.rank_computer = RankComputer(ranking_group_col, tuner_col)
    
    def compute_for_feature(
        self,
        data: pd.DataFrame,
        feature_name: str,
    ) -> dict[str, PartialDependenceResult]:
        """Compute partial dependence for one feature across all tuners.
        
        Args:
            data: Test data with features, ranking groups, and tuner identifiers
            feature_name: Name of feature to analyze
            
        Returns:
            Dictionary mapping tuner_name to PartialDependenceResult
        """
        if feature_name not in self.feature_cols:
            raise ValueError(f"Feature '{feature_name}' not in feature_cols")
        
        feature_values = data[feature_name]
        x_grid = self.grid_generator.generate_grid(feature_values)
        
        # Get unique tuners
        tuners = sorted(data[self.tuner_col].unique())
        
        # Storage for results
        tuner_pdp_results = {}
        
        for tuner in tuners:
            rank_means = []
            rank_stds = []
            
            for x_val in x_grid:
                # Create synthetic dataset with feature set to x_val
                synthetic_data = data.copy()
                synthetic_data[feature_name] = x_val
                
                # Predict scores
                dmatrix = xgb.DMatrix(synthetic_data[self.feature_cols])
                scores = self.model.predict(dmatrix)
                
                # Compute ranks by group
                tuner_ranks = self.rank_computer.compute_ranks_by_group(
                    synthetic_data, scores
                )
                
                # Extract ranks for this tuner
                if tuner in tuner_ranks:
                    all_ranks = []
                    for group_ranks in tuner_ranks[tuner].values():
                        all_ranks.extend(group_ranks)
                    
                    rank_means.append(np.mean(all_ranks))
                    rank_stds.append(np.std(all_ranks))
                else:
                    # Tuner not present in this configuration
                    rank_means.append(np.nan)
                    rank_stds.append(np.nan)
            
            # Count number of groups this tuner appears in
            tuner_data = data[data[self.tuner_col] == tuner]
            n_groups = tuner_data[self.ranking_group_col].nunique()
            
            tuner_pdp_results[tuner] = PartialDependenceResult(
                feature_name=feature_name,
                tuner_name=tuner,
                x_values=x_grid,
                rank_values=np.array(rank_means),
                rank_std=np.array(rank_stds),
                n_groups=n_groups,
            )
        
        return tuner_pdp_results
    
    def compute_for_all_features(
        self,
        data: pd.DataFrame,
        partition_name: str = "default",
    ) -> PartialDependenceResults:
        """Compute partial dependence for all features and tuners.
        
        Args:
            data: Test data with features, ranking groups, and tuner identifiers
            partition_name: Name of the data partition being analyzed
            
        Returns:
            PartialDependenceResults containing all PDP results
        """
        logger.info(f"Computing partial dependence for partition '{partition_name}'")
        logger.info(f"Features: {len(self.feature_cols)}, Tuners: {data[self.tuner_col].nunique()}")
        
        all_results = {}
        tuners = sorted(data[self.tuner_col].unique())
        
        for feature_name in self.feature_cols:
            logger.info(f"  Processing feature: {feature_name}")
            tuner_results = self.compute_for_feature(data, feature_name)
            
            for tuner, result in tuner_results.items():
                all_results[(feature_name, tuner)] = result
        
        return PartialDependenceResults(
            results=all_results,
            feature_names=self.feature_cols,
            tuner_names=tuners,
            partition_name=partition_name,
        )


class PartialDependencePlotter:
    """Creates visualizations for partial dependence results.
    
    Follows Single Responsibility Principle: only responsible for plotting.
    """
    
    def __init__(
        self,
        figsize_per_subplot: tuple[float, float] = (4, 3),
        n_cols: int = 3,
    ):
        """Initialize plotter.
        
        Args:
            figsize_per_subplot: Size of each subplot (width, height)
            n_cols: Number of columns in the plot grid
        """
        self.figsize_per_subplot = figsize_per_subplot
        self.n_cols = n_cols
    
    def plot_tuner_pdp_matrix(
        self,
        pdp_results: PartialDependenceResults,
        tuner_name: str,
        output_path: Path | None = None,
        show_std: bool = True,
    ) -> None:
        """Plot partial dependence matrix for a single tuner.
        
        Creates an n×3 grid where n = number of features, showing how each
        feature affects the tuner's rank across the partition.
        
        Args:
            pdp_results: Complete PDP results
            tuner_name: Name of tuner to plot
            output_path: Path to save plot (displays if None)
            show_std: Whether to show standard deviation bands
        """
        import matplotlib.pyplot as plt
        
        if tuner_name not in pdp_results.tuner_names:
            raise ValueError(f"Tuner '{tuner_name}' not found in results")
        
        # Get all results for this tuner
        tuner_results = [
            pdp_results.results[(feat, tuner_name)]
            for feat in pdp_results.feature_names
        ]
        
        # Calculate grid dimensions
        n_features = len(pdp_results.feature_names)
        n_rows = int(np.ceil(n_features / self.n_cols))
        
        # Create figure
        fig_width = self.figsize_per_subplot[0] * self.n_cols
        fig_height = self.figsize_per_subplot[1] * n_rows
        fig, axes = plt.subplots(n_rows, self.n_cols, figsize=(fig_width, fig_height))
        axes = np.atleast_2d(axes).flatten()
        
        # Plot each feature
        for idx, result in enumerate(tuner_results):
            ax = axes[idx]
            
            # Main line
            ax.plot(result.x_values, result.rank_values, 'o-', linewidth=2, markersize=4)
            
            # Standard deviation band
            if show_std:
                ax.fill_between(
                    result.x_values,
                    result.rank_values - result.rank_std,
                    result.rank_values + result.rank_std,
                    alpha=0.2,
                )
            
            ax.set_xlabel(result.feature_name, fontsize=9)
            ax.set_ylabel('Rank', fontsize=9)
            ax.set_title(f'{result.feature_name}\n(n_groups={result.n_groups})', fontsize=8)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=8)
            
            # Invert y-axis so lower rank (better) is at top
            ax.invert_yaxis()
        
        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].set_visible(False)
        
        fig.suptitle(
            f'Rank-Based Partial Dependence: {tuner_name}\n'
            f'Partition: {pdp_results.partition_name}',
            fontsize=12,
            fontweight='bold',
        )
        plt.tight_layout()
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_all_tuners(
        self,
        pdp_results: PartialDependenceResults,
        output_dir: Path,
        show_std: bool = True,
    ) -> None:
        """Plot partial dependence matrices for all tuners.
        
        Args:
            pdp_results: Complete PDP results
            output_dir: Directory to save plots
            show_std: Whether to show standard deviation bands
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Plotting PDPs for {len(pdp_results.tuner_names)} tuners")
        
        for tuner in pdp_results.tuner_names:
            output_path = output_dir / f'pdp_{tuner}.png'
            self.plot_tuner_pdp_matrix(
                pdp_results, tuner, output_path, show_std
            )
            logger.info(f"  Saved: {output_path}")


def compute_partial_dependence(
    model: xgb.Booster,
    test_data: pd.DataFrame,
    feature_cols: list[str],
    ranking_group_col: str = 'ranking_group',
    tuner_col: str = 'tuner',
    partition_name: str = 'default',
    n_grid_points: int = 20,
    quantile_range: tuple[float, float] = (0.05, 0.95),
) -> PartialDependenceResults:
    """Compute rank-based partial dependence for all features and tuners.
    
    This function computes how each feature affects the ranking position of each
    tuner (algorithm) within ranking groups. Unlike traditional partial dependence
    which measures score impact, this measures rank impact, aligning with the
    ShaRP framework's focus on rank-based explanations.
    
    Args:
        model: Trained XGBoost booster
        test_data: Test data with features, ranking groups, and tuner identifiers
        feature_cols: List of feature column names to analyze
        ranking_group_col: Column name for ranking groups
        tuner_col: Column name for tuner identifiers
        partition_name: Name of the data partition being analyzed
        n_grid_points: Number of grid points for continuous features
        quantile_range: (min, max) quantiles for continuous feature range
        
    Returns:
        PartialDependenceResults containing PDP for all features and tuners
        
    Example:
        >>> results = compute_partial_dependence(
        ...     model=trained_model,
        ...     test_data=test_df,
        ...     feature_cols=['feature1', 'feature2'],
        ...     partition_name='all_random'
        ... )
        >>> # Access specific result
        >>> pdp = results.results[('feature1', 'tuner_A')]
        >>> print(pdp.rank_values)
    """
    grid_generator = FeatureGridGenerator(
        n_grid_points=n_grid_points,
        quantile_range=quantile_range,
    )
    
    computer = PartialDependenceComputer(
        model=model,
        feature_cols=feature_cols,
        ranking_group_col=ranking_group_col,
        tuner_col=tuner_col,
        grid_generator=grid_generator,
    )
    
    return computer.compute_for_all_features(test_data, partition_name)


def plot_partial_dependence(
    pdp_results: PartialDependenceResults,
    output_dir: Path,
    tuner_name: str | None = None,
    show_std: bool = True,
    n_cols: int = 3,
) -> None:
    """Plot rank-based partial dependence results.
    
    Creates n×3 grid plots showing how features affect tuner rankings.
    
    Args:
        pdp_results: Results from compute_partial_dependence
        output_dir: Directory to save plots
        tuner_name: Specific tuner to plot (None plots all tuners)
        show_std: Whether to show standard deviation bands
        n_cols: Number of columns in the plot grid
        
    Example:
        >>> plot_partial_dependence(
        ...     pdp_results=results,
        ...     output_dir=Path('output/pdp'),
        ...     tuner_name='tuner_A'
        ... )
    """
    plotter = PartialDependencePlotter(n_cols=n_cols)
    
    if tuner_name:
        output_path = Path(output_dir) / f'pdp_{tuner_name}.png'
        plotter.plot_tuner_pdp_matrix(pdp_results, tuner_name, output_path, show_std)
    else:
        plotter.plot_all_tuners(pdp_results, output_dir, show_std)


def run_partial_dependence_analysis(
    models_by_partition: dict[str, xgb.Booster],
    test_data_by_partition: dict[str, pd.DataFrame],
    feature_cols: list[str],
    output_dir: Path | None = None,
    ranking_group_col: str = 'ranking_group',
    tuner_col: str = 'tuner',
    n_grid_points: int = 20,
    show_std: bool = True,
) -> dict[str, PartialDependenceResults]:
    """Run complete partial dependence analysis across multiple partitions.
    
    This is the main entry point for running PDP analysis on LTR models.
    It computes and visualizes rank-based partial dependence for each partition.
    
    Args:
        models_by_partition: Dictionary mapping partition names to trained models
        test_data_by_partition: Dictionary mapping partition names to test data
        feature_cols: List of feature column names to analyze
        output_dir: Directory to save outputs (None skips saving)
        ranking_group_col: Column name for ranking groups
        tuner_col: Column name for tuner identifiers
        n_grid_points: Number of grid points for continuous features
        show_std: Whether to show standard deviation bands in plots
        
    Returns:
        Dictionary mapping partition names to PartialDependenceResults
        
    Example:
        >>> from hpobench.learning_to_rank.pipeline import run_all_analyses
        >>> 
        >>> # Run LTR analysis
        >>> ltr_results = run_all_analyses(raw_data, schema, config)
        >>> 
        >>> # Extract models and test data
        >>> models = {name: res.model for name, res in ltr_results.items()}
        >>> test_data = {name: res.test_data for name, res in ltr_results.items()}
        >>> feature_cols = ltr_results['all_random'].feature_cols
        >>> 
        >>> # Run PDP analysis
        >>> pdp_results = run_partial_dependence_analysis(
        ...     models_by_partition=models,
        ...     test_data_by_partition=test_data,
        ...     feature_cols=feature_cols,
        ...     output_dir=Path('output/pdp_analysis')
        ... )
    """
    logger.info(f"Running partial dependence analysis for {len(models_by_partition)} partitions")
    
    all_results = {}
    
    for partition_name in models_by_partition.keys():
        logger.info(f"\nProcessing partition: {partition_name}")
        
        model = models_by_partition[partition_name]
        test_data = test_data_by_partition[partition_name]
        
        # Compute PDP
        pdp_results = compute_partial_dependence(
            model=model,
            test_data=test_data,
            feature_cols=feature_cols,
            ranking_group_col=ranking_group_col,
            tuner_col=tuner_col,
            partition_name=partition_name,
            n_grid_points=n_grid_points,
        )
        
        all_results[partition_name] = pdp_results
        
        # Plot if output directory provided
        if output_dir:
            partition_dir = Path(output_dir) / partition_name
            plot_partial_dependence(
                pdp_results=pdp_results,
                output_dir=partition_dir,
                show_std=show_std,
            )
    
    logger.info("\nPartial dependence analysis complete")
    return all_results
