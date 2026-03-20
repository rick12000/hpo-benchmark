import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from hpobench.config.schema import BenchmarkDataSchema
from hpobench.config.types import (
    LTRConfig,
    DownsamplingResults,
)
from hpobench.learning_to_rank.model import LTRModel, _evaluate_rankings

logger = logging.getLogger(__name__)


def compute_downsampling_curve(
    train_val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_cols: list[str],
    requested_sample_sizes: list[int],
    ltr_config: LTRConfig,
    schema: BenchmarkDataSchema,
    analysis_identifier: str,
) -> DownsamplingResults:
    """Train models at progressively smaller sizes and evaluate on fixed test set.

    At each checkpoint, a fresh model trains on a random subset of ranking groups.
    Validation split preserves original proportions. Models evaluated on fixed test set.

    Args:
        train_val_data: Combined train/validation data.
        test_data: Fixed held-out test set.
        feature_cols: Feature column names.
        requested_sample_sizes: Group counts to evaluate.
        ltr_config: LTR configuration.
        schema: Data schema.
        analysis_identifier: Logging label.

    Returns:
        DownsamplingResults with metric trajectories.
    """
    total_available_groups = train_val_data[schema.ranking_group_id_col].nunique()
    val_proportion = ltr_config.val_size / (ltr_config.train_size + ltr_config.val_size)
    valid_sample_sizes = sorted(
        {s for s in requested_sample_sizes if s <= total_available_groups} | {total_available_groups}
    )

    logger.info(
        f'[{analysis_identifier}] downsampling: {len(valid_sample_sizes)} checkpoints, '
        f'max {total_available_groups} groups'
    )

    all_group_ids = train_val_data[schema.ranking_group_id_col].unique()
    rng = np.random.RandomState(ltr_config.random_state)
    checkpoint_rows = []

    for n_groups_at_checkpoint in valid_sample_sizes:
        if n_groups_at_checkpoint < total_available_groups:
            sampled_group_ids = rng.choice(all_group_ids, size=n_groups_at_checkpoint, replace=False)
        else:
            sampled_group_ids = all_group_ids

        checkpoint_data = train_val_data[
            train_val_data[schema.ranking_group_id_col].isin(sampled_group_ids)
        ].copy()

        n_val_groups = max(1, int(n_groups_at_checkpoint * val_proportion))
        val_group_ids = rng.permutation(sampled_group_ids)[n_groups_at_checkpoint - n_val_groups:]
        checkpoint_train = checkpoint_data[
            ~checkpoint_data[schema.ranking_group_id_col].isin(val_group_ids)
        ].copy()
        checkpoint_val = checkpoint_data[
            checkpoint_data[schema.ranking_group_id_col].isin(val_group_ids)
        ].copy()

        checkpoint_model = LTRModel()
        checkpoint_model.fit(
            train_data=checkpoint_train,
            val_data=checkpoint_val,
            group_col=schema.ranking_group_id_col,
            label_col=schema.label_col,
            tuner_col=schema.tuner_col,
            feature_cols=feature_cols,
            tuning_config=ltr_config.tuning,
            k_values=ltr_config.k_values,
            analysis_identifier=f'{analysis_identifier}_downsampling_{n_groups_at_checkpoint}',
            n_tuning_trials=1,
        )

        checkpoint_metrics = _evaluate_rankings(
            test_data=test_data,
            predicted_scores=checkpoint_model.predict(test_data),
            k_values=ltr_config.k_values,
            ranking_group_id_col=schema.ranking_group_id_col,
            label_col=schema.label_col,
            tuner_col=schema.tuner_col,
        )
        checkpoint_rows.append({
            'sample_sizes': n_groups_at_checkpoint,
            'n_train_groups': n_groups_at_checkpoint - n_val_groups,
            'n_val_groups': n_val_groups,
            **checkpoint_metrics,
        })

    results_df = pd.DataFrame(checkpoint_rows)
    metric_trajectories = {
        col: results_df[col].tolist()
        for col in results_df.columns
        if col.startswith('precision@') or col.startswith('ndcg@')
    }

    return DownsamplingResults(
        sample_sizes=results_df['sample_sizes'].tolist(),
        n_train_groups=results_df['n_train_groups'].tolist(),
        n_val_groups=results_df['n_val_groups'].tolist(),
        metrics=metric_trajectories,
    )



def plot_downsampling_curve(
    downsampling_results: DownsamplingResults,
    output_path: Path,
    partition_name: str = '',
) -> None:
    """4-panel plot of precision@k and NDCG@k vs. training sample size.

    Args:
        downsampling_results: Downsampling results.
        output_path: Full path where PNG is saved.
        partition_name: Optional label for plot title.
    """
    sample_sizes = downsampling_results.sample_sizes
    relevant_metric_keys = [
        key for key in downsampling_results.metrics
        if key.startswith('precision@') or key.startswith('ndcg@')
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, metric_key in zip(axes, relevant_metric_keys[:4]):
        metric_values = downsampling_results.metrics[metric_key]
        ax.plot(sample_sizes, metric_values, 'o-', linewidth=2, markersize=6, color='#1f77b4')
        ax.set_xlabel('Training sample size (groups)', fontsize=11)
        ax.set_ylabel(metric_key.replace('@', ' @ ').title(), fontsize=11)
        ax.set_title(metric_key.replace('@', ' @ ').upper(), fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        if max(sample_sizes) / min(sample_sizes) > 10:
            ax.set_xscale('log')
        full_data_value = metric_values[-1]
        ax.axhline(
            full_data_value,
            color='red',
            linestyle='--',
            alpha=0.5,
            linewidth=1.5,
            label=f'Full data: {full_data_value:.3f}',
        )
        ax.legend(fontsize=9)

    for unused_ax in axes[len(relevant_metric_keys):]:
        unused_ax.set_visible(False)

    title = f'LTR Scaling Analysis\nPartition: {partition_name}' if partition_name else 'LTR Scaling Analysis'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Saved downsampling plot: {output_path}')
