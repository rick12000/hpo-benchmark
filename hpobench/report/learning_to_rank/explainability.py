"""
Feature importance and explainability for learning-to-rank models.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from dataclasses import dataclass


@dataclass
class SharpResults:
    """Results from ShaRP explainability analysis."""
    shap_values: np.ndarray
    feature_names: list[str]
    feature_matrix: np.ndarray
    base_value: float


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
