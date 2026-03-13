import logging
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from pathlib import Path

from hpobench.config.types import SharpResults, PartialDependenceResult, PartialDependenceResults
from sharp import ShaRP

logger = logging.getLogger(__name__)


def compute_shap_values(
    model: xgb.Booster,
    data: pd.DataFrame,
    feature_cols: list[str],
    sample_size: int | None = None,
    random_state: int = 42,
) -> SharpResults:
    """Compute rank-based SHAP values via ShaRP.

    Args:
        model: Trained XGBoost booster.
        data: Data to explain.
        feature_cols: Feature column names.
        sample_size: Perturbation sample size.
        random_state: Random seed.
    
    Returns:
        SharpResults with SHAP values and metadata.
    """
    X = data[feature_cols].values

    def predict(x: np.ndarray) -> np.ndarray:
        return model.predict(xgb.DMatrix(x, feature_names=feature_cols))

    explainer = ShaRP(
        qoi='rank',
        target_function=predict,
        measure='shapley',
        sample_size=sample_size,
        replace=False,
        random_state=random_state,
        n_jobs=1,
        verbose=0,
    )
    explainer.fit(X, feature_names=feature_cols)
    shap_values = explainer.all(X=X)

    return SharpResults(shap_values=shap_values, feature_names=feature_cols, feature_matrix=X)


def shap_importance_summary(shap_results: SharpResults) -> pd.DataFrame:
    """Compute per-feature importance from SHAP values.
    
    Args:
        shap_results: SHAP results from ShaRP.
    
    Returns:
        DataFrame sorted by mean absolute SHAP.
    """
    return (
        pd.DataFrame({
            'feature': shap_results.feature_names,
            'mean_abs_shap': np.abs(shap_results.shap_values).mean(axis=0),
            'mean_shap': shap_results.shap_values.mean(axis=0),
            'std_shap': shap_results.shap_values.std(axis=0),
        })
        .sort_values('mean_abs_shap', ascending=False)
        .reset_index(drop=True)
    )


def _save_figure(fig: plt.Figure, output_path: Path | None) -> None:
    """Save figure to disk or display."""
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_shap_importance(shap_results: SharpResults, output_path: Path | None = None, top_k: int = 20) -> None:
    """Horizontal bar chart of feature importance by mean |SHAP|.
    
    Args:
        shap_results: SHAP results.
        output_path: Save path or None to display.
        top_k: Number of top features to show.
    """
    importance = np.abs(shap_results.shap_values).mean(axis=0)
    idx = np.argsort(importance)[-top_k:][::-1]

    fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.3)))
    ax.barh(range(len(idx)), importance[idx], color='#ff0051')
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([shap_results.feature_names[i] for i in idx])
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title(f'Feature Importance (Top {top_k})')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    _save_figure(fig, output_path)


def plot_shap_beeswarm(shap_results: SharpResults, output_path: Path | None = None, top_k: int = 20) -> None:
    """Beeswarm plot of SHAP distributions per feature.
    
    Args:
        shap_results: SHAP results.
        output_path: Save path or None to display.
        top_k: Number of top features to show.
    """
    importance = np.abs(shap_results.shap_values).mean(axis=0)
    idx = np.argsort(importance)[-top_k:][::-1]
    rng = np.random.RandomState(42)

    fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.4)))
    for pos, feat_idx in enumerate(idx):
        shap_vals = shap_results.shap_values[:, feat_idx]
        feat_vals = shap_results.feature_matrix[:, feat_idx]
        vmin, vmax = feat_vals.min(), feat_vals.max()
        colors = (feat_vals - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(feat_vals)
        ax.scatter(
            shap_vals,
            pos + rng.uniform(-0.3, 0.3, len(shap_vals)),
            c=colors,
            cmap='coolwarm',
            s=20,
            alpha=0.6,
        )

    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([shap_results.feature_names[i] for i in idx])
    ax.set_xlabel('SHAP value')
    ax.set_title(f'Feature Impact Distribution (Top {top_k})')
    ax.axvline(0, color='black', linewidth=0.8, alpha=0.5)
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    _save_figure(fig, output_path)


def run_shap_analysis(
    model: xgb.Booster,
    data: pd.DataFrame,
    feature_cols: list[str],
    output_dir: Path | None = None,
    top_k: int = 20,
    sample_size: int | None = None,
) -> dict:
    """Compute SHAP values and save plots and summary.

    Args:
        model: Trained XGBoost booster.
        data: Data to explain.
        feature_cols: Feature column names.
        output_dir: Directory to save outputs.
        top_k: Top features to include.
        sample_size: ShaRP sample size.

    Returns:
        Dict with 'shap_results' and 'summary'.
    """
    shap_results = compute_shap_values(model=model, data=data, feature_cols=feature_cols, sample_size=sample_size)
    summary = shap_importance_summary(shap_results=shap_results)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_dir / 'feature_importance.csv', index=False)
        plot_shap_importance(shap_results=shap_results, output_path=output_dir / 'importance_bar.png', top_k=top_k)
        plot_shap_beeswarm(shap_results=shap_results, output_path=output_dir / 'importance_beeswarm.png', top_k=top_k)

    return {'shap_results': shap_results, 'summary': summary}


def _feature_grid(values: pd.Series, n_points: int, quantile_range: tuple[float, float]) -> np.ndarray:
    """Build evaluation grid for one feature.
    
    Categorical or low-cardinality: use unique values.
    Continuous: use quantile-bounded linspace to avoid extrapolation.
    
    Args:
        values: Feature values.
        n_points: Grid resolution.
        quantile_range: Quantile bounds.
    
    Returns:
        Grid points for the feature.
    """
    unique = values.dropna().unique()
    if len(unique) <= 10 or values.dtype == 'object':
        return np.sort(unique)
    lo, hi = values.quantile(quantile_range[0]), values.quantile(quantile_range[1])
    if lo == hi:
        return np.array([lo])
    return np.linspace(lo, hi, n_points)


def _bootstrap_ci(values: list[float], n_resamples: int, ci: float) -> tuple[float, float, float]:
    """Bootstrap confidence interval on mean rank.

    Args:
        values: One rank per ranking group.
        n_resamples: Bootstrap resamples.
        ci: CI coverage, e.g. 0.95.
    
    Returns:
        Tuple of (mean, ci_low, ci_high).
    """
    arr = np.array(values)
    rng = np.random.RandomState(42)
    boot_means = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_resamples)])
    tail = (1.0 - ci) / 2.0
    return float(arr.mean()), float(np.percentile(boot_means, 100 * tail)), float(np.percentile(boot_means, 100 * (1.0 - tail)))


def compute_partial_dependence(
    model: xgb.Booster,
    data: pd.DataFrame,
    feature_cols: list[str],
    group_col: str,
    tuner_col: str,
    partition_name: str = 'default',
    n_grid_points: int = 20,
    quantile_range: tuple[float, float] = (0.05, 0.95),
    n_bootstrap: int = 500,
    bootstrap_ci: float = 0.95,
) -> PartialDependenceResults:
    """Compute rank-based partial dependence for all features and tuners.
    
    Sweeps each feature across a grid while holding others at observed values.
    Bootstrapping over ranking groups provides non-parametric confidence bands.

    Args:
        model: Trained XGBoost booster.
        data: Test-set data with feature, group, and tuner columns.
        feature_cols: Meta-feature column names to sweep.
        group_col: Column identifying ranking groups.
        tuner_col: Column identifying tuners.
        partition_name: Label for downstream identification.
        n_grid_points: Grid resolution for continuous features.
        quantile_range: Quantile bounds for evaluation grid.
        n_bootstrap: Bootstrap resamples per grid point.
        bootstrap_ci: CI coverage, e.g. 0.95.

    Returns:
        PartialDependenceResults with one result per (feature, tuner) pair.
    """
    tuners = sorted(data[tuner_col].unique())
    n_groups = data[group_col].nunique()
    results: list[PartialDependenceResult] = []

    logger.info(f'Computing PDP for {partition_name!r} ({len(feature_cols)} features, {len(tuners)} tuners)')

    for feature in feature_cols:
        grid = _feature_grid(values=data[feature], n_points=n_grid_points, quantile_range=quantile_range)

        tuner_curves: dict[str, list[tuple[float, float, float]]] = {t: [] for t in tuners}

        for val in grid:
            modified = data.copy()
            modified[feature] = val
            modified['_score'] = model.predict(xgb.DMatrix(modified[feature_cols], feature_names=feature_cols))

            grid_point_ranks: dict[str, list[float]] = {t: [] for t in tuners}
            for _, group in modified.groupby(group_col):
                ranked = group['_score'].rank(ascending=False, method='first').astype(int)
                for tuner, rank in zip(group[tuner_col], ranked):
                    if tuner in grid_point_ranks:
                        grid_point_ranks[tuner].append(float(rank))

            for tuner in tuners:
                ranks = grid_point_ranks[tuner]
                tuner_curves[tuner].append(
                    _bootstrap_ci(values=ranks, n_resamples=n_bootstrap, ci=bootstrap_ci) if ranks else (np.nan, np.nan, np.nan)
                )

        for tuner in tuners:
            means, ci_lows, ci_highs = zip(*tuner_curves[tuner])
            results.append(PartialDependenceResult(
                feature_name=feature,
                tuner_name=tuner,
                x_values=grid,
                rank_means=np.array(means),
                rank_ci_lower=np.array(ci_lows),
                rank_ci_upper=np.array(ci_highs),
                n_groups=n_groups,
            ))

    return PartialDependenceResults(results=results, partition_name=partition_name)


def plot_partial_dependence(
    pdp_results: PartialDependenceResults,
    output_dir: Path,
    tuner_name: str | None = None,
    show_ci: bool = True,
    n_cols: int = 3,
) -> None:
    """Save per-tuner PDP grid plots.
    
    Each subplot shows mean rank with optional bootstrapped CI band.
    Y-axis inverted so rank 1 (best) appears at top.

    Args:
        pdp_results: Computed partial dependence results.
        output_dir: Directory to save PNG files.
        tuner_name: Plot only this tuner; None plots all.
        show_ci: Whether to draw bootstrapped CI band.
        n_cols: Subplot columns.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tuners = [tuner_name] if tuner_name is not None else pdp_results.tuner_names

    for tuner in tuners:
        if tuner not in pdp_results.tuner_names:
            raise ValueError(f'Tuner {tuner!r} not found in results')

        n_features = len(pdp_results.feature_names)
        n_rows = int(np.ceil(n_features / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        axes = np.atleast_2d(axes).flatten()

        for i, feature in enumerate(pdp_results.feature_names):
            r = pdp_results.get(feature=feature, tuner=tuner)
            ax = axes[i]
            ax.plot(r.x_values, r.rank_means, 'o-', linewidth=2, markersize=4)
            if show_ci:
                ax.fill_between(r.x_values, r.rank_ci_lower, r.rank_ci_upper, alpha=0.2)
            ax.set_xlabel(feature, fontsize=9)
            ax.set_ylabel('Mean rank', fontsize=9)
            ax.set_title(f'{feature}\n(n_groups={r.n_groups})', fontsize=8)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=8)
            ax.invert_yaxis()

        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)

        fig.suptitle(
            f'Rank-Based Partial Dependence: {tuner}\nPartition: {pdp_results.partition_name}',
            fontsize=12,
            fontweight='bold',
        )
        fig.tight_layout()
        path = output_dir / f'pdp_{tuner}.png'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
