import logging
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from pathlib import Path

from hpobench.config.types import (
    SharpResults,
    PartialDependenceResult,
    PartialDependenceResults,
    DownsamplingResults,
)
from sharp import ShaRP

logger = logging.getLogger(__name__)

def compute_shap_values(
    model: xgb.Booster,
    data: pd.DataFrame,
    feature_cols: list[str],
    sample_size: int | None = None,
    random_state: int = 42,
) -> SharpResults:
    """Compute rank-based SHAP values via the ShaRP library.

    Args:
        model: Trained XGBoost booster.
        data: Data to explain.
        feature_cols: Feature column names.
        sample_size: Perturbation sample size (``None`` → ShaRP default).
        random_state: Random seed.

    Raises:
        ImportError: If ``xai-sharp`` is not installed.
    """

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
    ranks = np.argsort(np.argsort(-score_fn(X))) + 1

    return SharpResults(
        shap_values=shap_values,
        feature_names=feature_cols,
        feature_matrix=X,
        base_value=float(ranks.mean()),
    )


def shap_importance_summary(results: SharpResults) -> pd.DataFrame:
    """Per-feature importance summary from SHAP values, sorted by mean |SHAP|."""
    return (
        pd.DataFrame({
            'feature': results.feature_names,
            'mean_abs_shap': np.abs(results.shap_values).mean(axis=0),
            'mean_shap': results.shap_values.mean(axis=0),
            'std_shap': results.shap_values.std(axis=0),
        })
        .sort_values('mean_abs_shap', ascending=False)
        .reset_index(drop=True)
    )


def _save_figure(fig: plt.Figure, output_path: Path | None) -> None:
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_shap_importance(
    results: SharpResults,
    output_path: Path | None = None,
    top_k: int = 20,
) -> None:
    """Horizontal bar chart of global feature importance (mean |SHAP|)."""
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
    fig.tight_layout()
    _save_figure(fig, output_path)


def plot_shap_beeswarm(
    results: SharpResults,
    output_path: Path | None = None,
    top_k: int = 20,
) -> None:
    """Beeswarm plot of SHAP value distributions per feature."""
    importance = np.abs(results.shap_values).mean(axis=0)
    top_indices = np.argsort(importance)[-top_k:][::-1]
    rng = np.random.RandomState(42)

    fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.4)))
    for i, feat_idx in enumerate(top_indices):
        shap_vals = results.shap_values[:, feat_idx]
        feat_vals = results.feature_matrix[:, feat_idx]
        vmin, vmax = feat_vals.min(), feat_vals.max()
        colors = (feat_vals - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(feat_vals)
        ax.scatter(shap_vals, i + rng.uniform(-0.3, 0.3, len(shap_vals)),
                   c=colors, cmap='coolwarm', s=20, alpha=0.6)

    ax.set_yticks(range(len(top_indices)))
    ax.set_yticklabels([results.feature_names[i] for i in top_indices])
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
    """Compute SHAP values, build summary, and optionally write plots and CSV.

    Returns:
        ``{'shap_results': SharpResults, 'summary': pd.DataFrame}``
    """
    shap_results = compute_shap_values(model, data, feature_cols, sample_size)
    summary = shap_importance_summary(shap_results)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_dir / 'feature_importance.csv', index=False)
        plot_shap_importance(shap_results, output_dir / 'importance_bar.png', top_k)
        plot_shap_beeswarm(shap_results, output_dir / 'importance_beeswarm.png', top_k)

    return {'shap_results': shap_results, 'summary': summary}


def _feature_grid(
    values: pd.Series,
    n_grid_points: int = 20,
    quantile_range: tuple[float, float] = (0.05, 0.95),
    categorical_threshold: int = 10,
) -> np.ndarray:
    """Return an evaluation grid for *values*.

    Low-cardinality and categorical features get all unique values.
    Continuous features get a quantile-bounded linspace.
    """
    unique = values.dropna().unique()
    if len(unique) <= categorical_threshold or values.dtype == 'object':
        return np.sort(unique)
    x_min, x_max = values.quantile(quantile_range[0]), values.quantile(quantile_range[1])
    if x_min == x_max:
        return np.array([x_min])
    return np.linspace(x_min, x_max, n_grid_points)


def _ranks_by_group(
    data: pd.DataFrame,
    scores: np.ndarray,
    ranking_group_col: str,
    tuner_col: str,
) -> dict[str, list[float]]:
    """Map each tuner to a flat list of its ranks across all ranking groups."""
    data = data.copy()
    data['_score'] = scores
    tuner_ranks: dict[str, list[float]] = {}

    for _, group_df in data.groupby(ranking_group_col):
        ranked = group_df['_score'].rank(ascending=False, method='first').astype(int)
        for tuner, rank in zip(group_df[tuner_col], ranked):
            tuner_ranks.setdefault(tuner, []).append(float(rank))

    return tuner_ranks


def compute_partial_dependence(
    model: xgb.Booster,
    data: pd.DataFrame,
    feature_cols: list[str],
    ranking_group_col: str,
    tuner_col: str,
    partition_name: str = 'default',
    n_grid_points: int = 20,
    quantile_range: tuple[float, float] = (0.05, 0.95),
) -> PartialDependenceResults:
    """Compute rank-based partial dependence for all features and tuners.

    For each feature, the feature value is swept across a grid while all
    other features are held at their observed values. Ranks are computed
    within each ranking group and averaged across groups per tuner.
    """
    tuners = sorted(data[tuner_col].unique())
    all_results: dict[tuple[str, str], PartialDependenceResult] = {}

    logger.info(
        f"Computing PDP for '{partition_name}' "
        f"({len(feature_cols)} features, {len(tuners)} tuners)"
    )

    for feature_name in feature_cols:
        logger.debug(f"  feature: {feature_name}")
        x_grid = _feature_grid(data[feature_name], n_grid_points, quantile_range)

        rank_lists: dict[str, list[list[float]]] = {t: [[] for _ in x_grid] for t in tuners}

        for grid_idx, x_val in enumerate(x_grid):
            synthetic = data.copy()
            synthetic[feature_name] = x_val
            scores = model.predict(xgb.DMatrix(synthetic[feature_cols]))
            for tuner, ranks in _ranks_by_group(synthetic, scores, ranking_group_col, tuner_col).items():
                if tuner in rank_lists:
                    rank_lists[tuner][grid_idx].extend(ranks)

        for tuner in tuners:
            means = [np.mean(r) if r else np.nan for r in rank_lists[tuner]]
            stds  = [np.std(r)  if r else np.nan for r in rank_lists[tuner]]
            n_groups = data[data[tuner_col] == tuner][ranking_group_col].nunique()
            all_results[(feature_name, tuner)] = PartialDependenceResult(
                feature_name=feature_name,
                tuner_name=tuner,
                x_values=x_grid,
                rank_values=np.array(means),
                rank_std=np.array(stds),
                n_groups=n_groups,
            )

    return PartialDependenceResults(
        results=all_results,
        feature_names=feature_cols,
        tuner_names=tuners,
        partition_name=partition_name,
    )


def plot_partial_dependence(
    pdp_results: PartialDependenceResults,
    output_dir: Path,
    tuner_name: str | None = None,
    show_std: bool = True,
    n_cols: int = 3,
) -> None:
    """Save per-tuner PDP grid plots to *output_dir*."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tuners = [tuner_name] if tuner_name else pdp_results.tuner_names

    for tuner in tuners:
        if tuner not in pdp_results.tuner_names:
            raise ValueError(f"Tuner '{tuner}' not found in results")

        n_features = len(pdp_results.feature_names)
        n_rows = int(np.ceil(n_features / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        axes = np.atleast_2d(axes).flatten()

        for idx, feature_name in enumerate(pdp_results.feature_names):
            result = pdp_results.results[(feature_name, tuner)]
            ax = axes[idx]
            ax.plot(result.x_values, result.rank_values, 'o-', linewidth=2, markersize=4)
            if show_std:
                ax.fill_between(
                    result.x_values,
                    result.rank_values - result.rank_std,
                    result.rank_values + result.rank_std,
                    alpha=0.2,
                )
            ax.set_xlabel(feature_name, fontsize=9)
            ax.set_ylabel('Rank', fontsize=9)
            ax.set_title(f'{feature_name}\n(n_groups={result.n_groups})', fontsize=8)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=8)
            ax.invert_yaxis()

        for idx in range(n_features, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(
            f'Rank-Based Partial Dependence: {tuner}\nPartition: {pdp_results.partition_name}',
            fontsize=12, fontweight='bold',
        )
        fig.tight_layout()
        path = output_dir / f'pdp_{tuner}.png'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.debug(f"Saved PDP plot: {path}")


def plot_downsampling_curve(
    results: DownsamplingResults,
    output_path: Path,
    partition_name: str = '',
) -> None:
    """4-panel plot of precision@k and NDCG@k vs. training sample size."""
    sample_sizes = results.sample_sizes
    metric_keys = [k for k in results.metrics if k.startswith('precision@') or k.startswith('ndcg@')]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, metric in zip(axes, metric_keys[:4]):
        values = results.metrics[metric]
        ax.plot(sample_sizes, values, 'o-', linewidth=2, markersize=6, color='#1f77b4')
        ax.set_xlabel('Training sample size (groups)', fontsize=11)
        ax.set_ylabel(metric.replace('@', ' @ ').title(), fontsize=11)
        ax.set_title(metric.replace('@', ' @ ').upper(), fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        if max(sample_sizes) / min(sample_sizes) > 10:
            ax.set_xscale('log')
        final = values[-1]
        ax.axhline(final, color='red', linestyle='--', alpha=0.5, linewidth=1.5,
                   label=f'Full data: {final:.3f}')
        ax.legend(fontsize=9)

    for ax in axes[len(metric_keys):]:
        ax.set_visible(False)

    title = f'LTR Scaling Analysis\nPartition: {partition_name}' if partition_name else 'LTR Scaling Analysis'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved downsampling plot: {output_path}")
