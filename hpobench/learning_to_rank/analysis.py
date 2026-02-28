"""
LTRAnalysis: core class for a single learning-to-rank experiment.

Lifecycle::

    analysis = LTRAnalysis(config, schema, partition='all', strategy='random')
    analysis.fit(raw_data)      # prepare data, train models
    analysis.evaluate()          # compute test-set metrics

    pdp   = analysis.compute_pdp(output_dir=Path('out/pdp'))
    shap  = analysis.compute_shap(output_dir=Path('out/shap'))
    curve = analysis.compute_downsampling(output_dir=Path('out/ds'))

    analysis.save(Path('out/results'))
"""

import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Literal

from hpobench.config.schema import BenchmarkDataSchema
from hpobench.learning_to_rank.model import (
    LTRConfig, NaiveRanker, LTRModel, evaluate_ltr, evaluate_naive,
)
from hpobench.learning_to_rank.preprocessing import prepare_data, split_data
from hpobench.learning_to_rank.explainability import (
    PartialDependenceResults,
    compute_partial_dependence,
    plot_partial_dependence,
    run_shap_analysis,
)

logger = logging.getLogger(__name__)


class LTRAnalysis:
    """Learning-to-rank experiment for a single (partition, strategy) pair."""

    def __init__(
        self,
        config: LTRConfig,
        schema: BenchmarkDataSchema,
        partition: Literal['all', 'synthetic', 'real'] = 'all',
        strategy: Literal['random', 'synthetic_train_real_test'] = 'random',
        tuner_encoding_method: Literal['ordinal', 'one_hot'] = 'ordinal',
        name: str = '',
    ) -> None:
        self.config = config
        self.schema = schema
        self.partition = partition
        self.strategy = strategy
        self.tuner_encoding_method = tuner_encoding_method
        self.name = name

        self._ltr_model: LTRModel | None = None
        self._naive_ranker: NaiveRanker | None = None
        self._train_data: pd.DataFrame | None = None
        self._val_data: pd.DataFrame | None = None
        self._test_data: pd.DataFrame | None = None
        self._feature_cols: list[str] = []
        self._ltr_metrics: dict[str, float] = {}
        self._naive_metrics: dict[str, float] = {}

    @property
    def is_fitted(self) -> bool:
        return self._ltr_model is not None

    @property
    def is_evaluated(self) -> bool:
        return bool(self._ltr_metrics)

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(f"LTRAnalysis '{self.name}': call fit() before this method.")

    def _require_evaluated(self) -> None:
        if not self.is_evaluated:
            raise RuntimeError(
                f"LTRAnalysis '{self.name}': call evaluate() after fit() before this method."
            )

    def fit(self, raw_data: pd.DataFrame) -> "LTRAnalysis":
        """Prepare data and train the LTR and naive models."""
        data, feature_cols = prepare_data(
            raw_data, self.schema, self.partition, self.tuner_encoding_method
        )
        train_data, val_data, test_data = split_data(data, self.strategy, self.config)

        self._feature_cols = feature_cols
        self._train_data = train_data
        self._val_data = val_data
        self._test_data = test_data

        self._naive_ranker = NaiveRanker().fit(
            train_data, self.schema.tuner_col, self.schema.label_col
        )
        self._ltr_model = LTRModel(self.config.xgb_params).fit(
            train_data, val_data, feature_cols,
            self.schema.label_col, self.schema.ranking_group_col,
        )
        logger.info(
            f"[{self.name}] fit – train={len(train_data)}, "
            f"val={len(val_data)}, test={len(test_data)}"
        )
        return self

    def evaluate(self) -> "LTRAnalysis":
        """Compute precision@k and NDCG@k on the held-out test set."""
        self._require_fitted()
        self._ltr_metrics = evaluate_ltr(
            self._ltr_model, self._test_data, self.config.k_values, self.schema
        )
        self._naive_metrics = evaluate_naive(
            self._naive_ranker, self._test_data, self.config.k_values, self.schema
        )
        p1 = self._ltr_metrics['precision@1']
        delta = p1 - self._naive_metrics['precision@1']
        logger.info(f"[{self.name}] evaluate – P@1={p1:.3f} ({delta:+.3f} vs naive)")
        return self

    def compute_pdp(
        self,
        output_dir: Path | None = None,
        n_grid_points: int = 20,
        show_std: bool = True,
    ) -> PartialDependenceResults:
        """Compute rank-based partial dependence plots."""
        self._require_fitted()
        pdp = compute_partial_dependence(
            model=self._ltr_model.booster,
            data=self._test_data,
            feature_cols=self._feature_cols,
            ranking_group_col=self.schema.ranking_group_col,
            tuner_col=self.schema.tuner_col,
            partition_name=self.name,
            n_grid_points=n_grid_points,
        )
        if output_dir is not None:
            plot_partial_dependence(pdp, Path(output_dir), show_std=show_std)
        return pdp

    def compute_shap(
        self,
        output_dir: Path | None = None,
        top_k: int = 20,
        sample_size: int | None = None,
    ) -> dict:
        """Compute ShaRP/SHAP feature attributions.

        Returns:
            ``{'shap_results': SharpResults, 'summary': pd.DataFrame}``
        """
        self._require_fitted()
        return run_shap_analysis(
            model=self._ltr_model.booster,
            data=self._test_data,
            feature_cols=self._feature_cols,
            output_dir=output_dir,
            top_k=top_k,
            sample_size=sample_size,
        )

    def compute_downsampling(
        self,
        output_dir: Path | None = None,
        sample_sizes: list[int] | None = None,
    ) -> dict[str, list]:
        """Evaluate how model performance scales with training data size.

        Re-trains at each sample size on the existing splits; the test set
        stays fixed throughout.

        Returns:
            Dict with 'sample_sizes', 'n_train_groups', 'n_val_groups',
            and one list per metric key.
        """
        self._require_fitted()

        train_val = pd.concat([self._train_data, self._val_data], ignore_index=True)
        n_total = train_val[self.schema.ranking_group_col].nunique()
        val_prop = self.config.val_size / (self.config.train_size + self.config.val_size)

        if sample_sizes is None:
            sample_sizes = _logarithmic_sizes(n_total)
        else:
            sample_sizes = sorted({s for s in sample_sizes if s <= n_total} | {n_total})

        logger.info(f"Downsampling: {len(sample_sizes)} checkpoints, max {n_total} groups")

        results: dict[str, list] = {'sample_sizes': [], 'n_train_groups': [], 'n_val_groups': []}
        for k in self.config.k_values:
            results[f'precision@{k}'] = []
            results[f'ndcg@{k}'] = []

        all_groups = train_val[self.schema.ranking_group_col].unique()
        rng = np.random.RandomState(self.config.random_state)

        for n_groups in sample_sizes:
            groups = rng.choice(all_groups, size=n_groups, replace=False) if n_groups < n_total else all_groups
            subset = train_val[train_val[self.schema.ranking_group_col].isin(groups)].copy()

            shuffled = rng.permutation(groups)
            n_val_groups = max(1, int(n_groups * val_prop))
            val_groups = shuffled[n_groups - n_val_groups:]
            sub_train = subset[~subset[self.schema.ranking_group_col].isin(val_groups)].copy()
            sub_val   = subset[ subset[self.schema.ranking_group_col].isin(val_groups)].copy()

            model = LTRModel(self.config.xgb_params).fit(
                sub_train, sub_val, self._feature_cols,
                self.schema.label_col, self.schema.ranking_group_col,
            )
            metrics = evaluate_ltr(model, self._test_data, self.config.k_values, self.schema)

            results['sample_sizes'].append(n_groups)
            results['n_train_groups'].append(n_groups - n_val_groups)
            results['n_val_groups'].append(n_val_groups)
            for k in self.config.k_values:
                results[f'precision@{k}'].append(metrics[f'precision@{k}'])
                results[f'ndcg@{k}'].append(metrics[f'ndcg@{k}'])

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(results).to_csv(output_dir / 'downsampling_curve.csv', index=False)
            self._plot_downsampling_curve(results, output_dir / 'downsampling_curve.png')

        return results

    def save(self, output_dir: Path) -> None:
        """Write test_data.csv and metrics.json to output_dir."""
        self._require_fitted()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._test_data.to_csv(output_dir / 'test_data.csv', index=False)
        payload: dict = {
            'n_train': len(self._train_data),
            'n_val': len(self._val_data),
            'n_test': len(self._test_data),
        }
        if self.is_evaluated:
            payload['ltr_metrics'] = self._ltr_metrics
            payload['naive_metrics'] = self._naive_metrics
        with open(output_dir / 'metrics.json', 'w') as fh:
            json.dump(payload, fh, indent=2)

    def summary(self) -> dict:
        """Flat dict suitable for a single row of a summary DataFrame."""
        self._require_evaluated()
        row = {
            'config': self.name,
            'partition': self.partition,
            'strategy': self.strategy,
            'n_train': len(self._train_data),
            'n_val': len(self._val_data),
            'n_test': len(self._test_data),
        }
        for k, v in self._ltr_metrics.items():
            row[f'{k}_ltr'] = v
        for k, v in self._naive_metrics.items():
            row[f'{k}_naive'] = v
        return row

    def _plot_downsampling_curve(self, results: dict[str, list], output_path: Path) -> None:
        """4-panel plot of precision@k and NDCG@k vs. training sample size."""
        sample_sizes = results['sample_sizes']
        metric_keys = [k for k in results if k.startswith('precision@') or k.startswith('ndcg@')]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for ax, metric in zip(axes, metric_keys[:4]):
            values = results[metric]
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

        fig.suptitle(f'LTR Scaling Analysis\nPartition: {self.name}', fontsize=14, fontweight='bold')
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved downsampling plot: {output_path}")


def _logarithmic_sizes(n_total: int) -> list[int]:
    """Logarithmically spaced integers from 10 up to n_total (inclusive)."""
    sizes: list[int] = []
    n = 10
    while n < n_total:
        sizes.append(n)
        n = int(n * 2) if n < 100 or n >= 1000 else int(n * 2.5)
    sizes.append(n_total)
    return sizes
