import pandas as pd
import numpy as np
import logging
import xgboost as xgb
from pathlib import Path
from hpobench.config.schema import BenchmarkDataSchema

logger = logging.getLogger(__name__)


def create_xgboost_score_function(
    model: xgb.Booster,
    feature_cols: list[str],
):
    """Create a scoring function wrapper for XGBoost model.
    
    Args:
        model: Trained XGBoost booster
        feature_cols: List of feature column names
        
    Returns:
        Function that takes feature matrix and returns scores
    """
    def score_function(X: np.ndarray) -> np.ndarray:
        """Score function for ShaRP explainability.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Array of scores of shape (n_samples,)
        """
        dmatrix = xgb.DMatrix(X)
        scores = model.predict(dmatrix)
        return scores
    
    return score_function


def compute_sharp_explanations(
    model: xgb.Booster,
    test_data: pd.DataFrame,
    feature_cols: list[str],
    qoi: str = "rank",
    sample_size: int | None = None,
    random_state: int = 42,
    n_jobs: int = 1,
    schema: BenchmarkDataSchema | None = None,
) -> dict:
    """Compute ShaRP explanations for learning-to-rank model.
    
    Args:
        model: Trained XGBoost model
        test_data: Test data with features and rankings
        feature_cols: List of feature column names
        qoi: Quantity of interest ("rank", "rank_score", or "top_k")
        sample_size: Number of samples for perturbation (None uses all)
        random_state: Random seed for reproducibility
        n_jobs: Number of parallel jobs
        schema: Optional BenchmarkDataSchema for column naming
        
    Returns:
        Dictionary containing:
            - sharp_values: Array of SHAP values (n_samples, n_features)
            - feature_names: List of feature names
            - feature_matrix: Feature matrix used for explanations
            - base_value: Mean rank in the reference dataset
    """
    try:
        from sharp import ShaRP
    except ImportError:
        raise ImportError(
            "ShaRP package not installed. Install with: pip install xai-sharp"
        )
    
    if schema is None:
        schema = BenchmarkDataSchema()
    
    logger.info(f"Computing ShaRP explanations with QoI: {qoi}")
    
    X_test = test_data[feature_cols].values
    feature_names = feature_cols
    
    logger.info(f"Test set size: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    score_function = create_xgboost_score_function(model, feature_cols)
    
    logger.info("Initializing ShaRP explainer")
    explainer = ShaRP(
        qoi=qoi,
        target_function=score_function,
        measure="shapley",
        sample_size=sample_size,
        replace=False,
        random_state=random_state,
        cache=True,
        n_jobs=n_jobs,
        verbose=1,
    )
    
    logger.info("Fitting ShaRP explainer on test data")
    explainer.fit(X_test, feature_names=feature_names)
    
    logger.info("Computing SHAP values for all test samples")
    sharp_values = explainer.all(X=X_test)
    
    scores = score_function(X_test)
    ranks = np.argsort(np.argsort(-scores)) + 1
    base_value = ranks.mean()
    
    logger.info(f"Computed SHAP values for {sharp_values.shape[0]} samples")
    logger.info(f"Mean rank (base value): {base_value:.2f}")
    
    return {
        'sharp_values': sharp_values,
        'feature_names': feature_names,
        'feature_matrix': X_test,
        'base_value': base_value,
        'explainer': explainer,
    }


def create_global_importance_plot(
    sharp_values: np.ndarray,
    feature_names: list[str],
    output_path: Path | None = None,
    top_k: int = 20,
) -> None:
    """Create waterfall plot of global feature importance.
    
    Args:
        sharp_values: Array of SHAP values (n_samples, n_features)
        feature_names: List of feature names
        output_path: Optional path to save the plot
        top_k: Number of top features to display
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting")
    
    logger.info("Creating global feature importance waterfall plot")
    
    mean_abs_sharp = np.abs(sharp_values).mean(axis=0)
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_sharp,
    })
    feature_importance_df = feature_importance_df.sort_values(
        'importance', 
        ascending=False
    ).head(top_k)
    
    plt.figure(figsize=(10, max(6, top_k * 0.3)))
    
    colors = ['#ff0051' if x > 0 else '#008bfb' for x in feature_importance_df['importance']]
    
    plt.barh(
        range(len(feature_importance_df)),
        feature_importance_df['importance'],
        color=colors,
    )
    
    plt.yticks(
        range(len(feature_importance_df)),
        feature_importance_df['feature'],
        fontsize=10,
    )
    plt.xlabel('Mean |SHAP value| (average impact on rank)', fontsize=12)
    plt.title(f'Global Feature Importance (Top {top_k})', fontsize=14, pad=20)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved global importance plot to {output_path}")
    
    plt.close()


def create_beeswarm_plot(
    sharp_values: np.ndarray,
    feature_matrix: np.ndarray,
    feature_names: list[str],
    output_path: Path | None = None,
    top_k: int = 20,
) -> None:
    """Create beeswarm plot showing feature value distributions and SHAP values.
    
    Args:
        sharp_values: Array of SHAP values (n_samples, n_features)
        feature_matrix: Feature matrix (n_samples, n_features)
        feature_names: List of feature names
        output_path: Optional path to save the plot
        top_k: Number of top features to display
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm
    except ImportError:
        raise ImportError("matplotlib required for plotting")
    
    logger.info("Creating beeswarm plot")
    
    mean_abs_sharp = np.abs(sharp_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_sharp)[-top_k:][::-1]
    
    fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.4)))
    
    for i, feat_idx in enumerate(top_indices):
        shap_vals = sharp_values[:, feat_idx]
        feat_vals = feature_matrix[:, feat_idx]
        
        feat_min = feat_vals.min()
        feat_max = feat_vals.max()
        feat_range = feat_max - feat_min
        
        if feat_range > 0:
            normalized_feat_vals = (feat_vals - feat_min) / feat_range
        else:
            normalized_feat_vals = np.zeros_like(feat_vals)
        
        y_positions = np.full_like(shap_vals, i, dtype=float)
        
        jitter_amount = 0.3
        y_jitter = np.random.RandomState(42 + i).uniform(
            -jitter_amount, 
            jitter_amount, 
            size=len(shap_vals)
        )
        y_positions = y_positions + y_jitter
        
        scatter = ax.scatter(
            shap_vals,
            y_positions,
            c=normalized_feat_vals,
            cmap='coolwarm',
            s=20,
            alpha=0.6,
            edgecolors='none',
        )
    
    ax.set_yticks(range(len(top_indices)))
    ax.set_yticklabels([feature_names[idx] for idx in top_indices], fontsize=10)
    ax.set_xlabel('SHAP value (impact on rank)', fontsize=12)
    ax.set_title(f'Feature Impact Distribution (Top {top_k})', fontsize=14, pad=20)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.grid(axis='x', alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Feature value\n(low to high)', fontsize=10)
    
    plt.tight_layout()
    
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved beeswarm plot to {output_path}")
    
    plt.close()


def create_feature_importance_summary(
    sharp_values: np.ndarray,
    feature_names: list[str],
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Create summary table of feature importance statistics.
    
    Args:
        sharp_values: Array of SHAP values (n_samples, n_features)
        feature_names: List of feature names
        output_path: Optional path to save the CSV
        
    Returns:
        DataFrame with feature importance statistics
    """
    logger.info("Creating feature importance summary")
    
    summary_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': np.abs(sharp_values).mean(axis=0),
        'mean_shap': sharp_values.mean(axis=0),
        'std_shap': sharp_values.std(axis=0),
        'min_shap': sharp_values.min(axis=0),
        'max_shap': sharp_values.max(axis=0),
    })
    
    summary_df = summary_df.sort_values('mean_abs_shap', ascending=False)
    
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_path, index=False)
        logger.info(f"Saved feature importance summary to {output_path}")
    
    logger.info("\nTop 10 Most Important Features:")
    for idx, row in summary_df.head(10).iterrows():
        logger.info(
            f"  {row['feature']}: "
            f"mean_abs={row['mean_abs_shap']:.4f}, "
            f"mean={row['mean_shap']:.4f}"
        )
    
    return summary_df


def run_sharp_analysis(
    model: xgb.Booster,
    test_data: pd.DataFrame,
    feature_cols: list[str],
    output_dir: Path | None = None,
    qoi: str = "rank",
    sample_size: int | None = None,
    random_state: int = 42,
    n_jobs: int = 1,
    top_k_features: int = 20,
    schema: BenchmarkDataSchema | None = None,
) -> dict:
    """Run complete ShaRP explainability analysis.
    
    Args:
        model: Trained XGBoost model
        test_data: Test data with features and rankings
        feature_cols: List of feature column names
        output_dir: Optional directory to save plots and results
        qoi: Quantity of interest ("rank", "rank_score", or "top_k")
        sample_size: Number of samples for perturbation (None uses all)
        random_state: Random seed for reproducibility
        n_jobs: Number of parallel jobs
        top_k_features: Number of top features to display in plots
        schema: Optional BenchmarkDataSchema for column naming
        
    Returns:
        Dictionary containing:
            - sharp_results: Results from compute_sharp_explanations
            - importance_summary: Feature importance summary DataFrame
    """
    logger.info("Starting ShaRP explainability analysis")
    
    sharp_results = compute_sharp_explanations(
        model=model,
        test_data=test_data,
        feature_cols=feature_cols,
        qoi=qoi,
        sample_size=sample_size,
        random_state=random_state,
        n_jobs=n_jobs,
        schema=schema,
    )
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        
        create_global_importance_plot(
            sharp_values=sharp_results['sharp_values'],
            feature_names=sharp_results['feature_names'],
            output_path=output_dir / 'global_feature_importance.png',
            top_k=top_k_features,
        )
        
        create_beeswarm_plot(
            sharp_values=sharp_results['sharp_values'],
            feature_matrix=sharp_results['feature_matrix'],
            feature_names=sharp_results['feature_names'],
            output_path=output_dir / 'feature_beeswarm.png',
            top_k=top_k_features,
        )
        
        importance_summary = create_feature_importance_summary(
            sharp_values=sharp_results['sharp_values'],
            feature_names=sharp_results['feature_names'],
            output_path=output_dir / 'feature_importance_summary.csv',
        )
    else:
        importance_summary = create_feature_importance_summary(
            sharp_values=sharp_results['sharp_values'],
            feature_names=sharp_results['feature_names'],
        )
    
    logger.info("ShaRP explainability analysis completed")
    
    return {
        'sharp_results': sharp_results,
        'importance_summary': importance_summary,
    }
