import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from hpobench.config.schema import BenchmarkDataSchema, SurrogateMetafeaturesSchema
from hpobench.config.constants import SyntheticGenerationParameters
from hpobench.config.types import (
    LTRConfig,
    Partition,
    SplitStrategy,
    TunerEncoding,
    PartialDependenceResults,
    DownsamplingResults,
)
from hpobench.learning_to_rank.model import NaiveRanker, LTRModel
from hpobench.learning_to_rank.preprocessing import prepare_data, split_data
from hpobench.learning_to_rank.explainability import (
    compute_partial_dependence,
    plot_partial_dependence,
    plot_downsampling_curve,
    run_shap_analysis,
)

logger = logging.getLogger(__name__)


def _precision_at_k(pred_ranking: list, true_ranking: list, k: int) -> float:
    return len(set(pred_ranking[:k]) & set(true_ranking[:k])) / k


def _evaluate_rankings(
    test_data: pd.DataFrame,
    predicted_scores: np.ndarray,
    k_values: tuple[int, ...],
    schema: BenchmarkDataSchema,
    ascending_scores: bool = False,
) -> dict[str, float]:
    """Compute precision@k and NDCG@k across all ranking groups."""
    from sklearn.metrics import ndcg_score

    test_data = test_data.copy()
    test_data['_pred_score'] = predicted_scores

    precision_lists = {k: [] for k in k_values}
    ndcg_lists = {k: [] for k in k_values}

    for _, group in test_data.groupby(schema.ranking_group_col):
        true_sorted = group.sort_values(schema.label_col, ascending=False)
        pred_sorted = group.sort_values('_pred_score', ascending=ascending_scores)

        true_ranking = true_sorted[schema.tuner_col].tolist()
        pred_ranking = pred_sorted[schema.tuner_col].tolist()

        true_relevance = np.arange(len(true_sorted), 0, -1)
        relevance_map = dict(zip(true_sorted[schema.tuner_col], true_relevance))
        pred_relevance = np.array([relevance_map[t] for t in pred_ranking])

        for k in k_values:
            precision_lists[k].append(_precision_at_k(pred_ranking, true_ranking, k))
            ndcg_lists[k].append(ndcg_score([true_relevance], [pred_relevance], k=k))

    return {
        **{f'precision@{k}': float(np.mean(precision_lists[k])) for k in k_values},
        **{f'ndcg@{k}': float(np.mean(ndcg_lists[k])) for k in k_values},
    }




class LTRAnalysis:
    """Single LTR experiment with mutable state for fit results."""

    def __init__(
        self,
        schema: BenchmarkDataSchema,
        partition: Partition = 'all',
        strategy: SplitStrategy = 'random',
        tuner_encoding_method: TunerEncoding = 'ordinal',
        analysis_identifier: str = '',
    ) -> None:
        self.config = LTRConfig()
        self.schema = schema
        self.partition = partition
        self.strategy = strategy
        self.tuner_encoding_method = tuner_encoding_method
        self.analysis_identifier = analysis_identifier

        self.train_data: pd.DataFrame | None = None
        self.val_data: pd.DataFrame | None = None
        self.test_data: pd.DataFrame | None = None
        self.feature_cols: list[str] = []
        self.ltr_model: LTRModel | None = None
        self.naive_ranker: NaiveRanker | None = None
        self.ltr_metrics: dict[str, float] = {}
        self.naive_metrics: dict[str, float] = {}

    def fit(self, raw_data: pd.DataFrame) -> "LTRAnalysis":
        synthetic_params = SyntheticGenerationParameters()
        metafeatures_schema = SurrogateMetafeaturesSchema()

        # Validate strategy eligibility
        if self.strategy == 'synthetic_train_real_test':
            if self.partition != 'all':
                raise ValueError(
                    f"Strategy 'synthetic_train_real_test' requires partition='all' "
                    f"but got partition='{self.partition}' for config '{self.analysis_identifier}'"
                )
            
            has_synthetic = (raw_data['benchmark_identifier'] == synthetic_params.benchmark_identifier).any()
            has_real = (raw_data['benchmark_identifier'] != synthetic_params.benchmark_identifier).any()
            
            if not has_synthetic:
                raise ValueError(
                    f"Strategy 'synthetic_train_real_test' requires synthetic data "
                    f"but no synthetic rows found in config '{self.analysis_identifier}'"
                )
            if not has_real:
                raise ValueError(
                    f"Strategy 'synthetic_train_real_test' requires real data for testing "
                    f"but no real rows found in config '{self.analysis_identifier}'"
                )

        data, feature_cols = prepare_data(
            raw_data=raw_data,
            partition=self.partition,
            schema=self.schema,
            metafeatures_schema=metafeatures_schema,
            synthetic_benchmark_id=synthetic_params.benchmark_identifier,
            tuner_encoding_method=self.tuner_encoding_method,
        )
        self.train_data, self.val_data, self.test_data = split_data(
            data=data,
            strategy=self.strategy,
            config=self.config,
            synthetic_benchmark_id=synthetic_params.benchmark_identifier,
        )
        self.feature_cols = feature_cols

        self.ltr_model = LTRModel(self.config.xgb_params, self.config.num_boost_rounds).fit(
            self.train_data, self.schema, self.feature_cols,
        )
        self.naive_ranker = NaiveRanker().fit(
            self.train_data, self.schema, self.feature_cols,
        )

        logger.info(
            f"[{self.analysis_identifier}] fit – train={len(self.train_data)}, "
            f"val={len(self.val_data)}, test={len(self.test_data)}"
        )
        return self

    def evaluate(self) -> "LTRAnalysis":
        if self.ltr_model is None or self.naive_ranker is None or self.test_data is None:
            raise RuntimeError(f"LTRAnalysis '{self.analysis_identifier}': call fit() before evaluate().")

        self.ltr_metrics = _evaluate_rankings(
            self.test_data, self.ltr_model.predict(self.test_data),
            self.config.k_values, self.schema, ascending_scores=False,
        )
        self.naive_metrics = _evaluate_rankings(
            self.test_data, self.naive_ranker.predict(self.test_data),
            self.config.k_values, self.schema, ascending_scores=True,
        )
        p1 = self.ltr_metrics['precision@1']
        delta = p1 - self.naive_metrics['precision@1']
        logger.info(f"[{self.analysis_identifier}] evaluate – P@1={p1:.3f} ({delta:+.3f} vs naive)")
        return self

    def compute_pdp(
        self,
        output_dir: Path | None = None,
        n_grid_points: int = 20,
        show_std: bool = True,
    ) -> PartialDependenceResults:
        if self.ltr_model is None or self.test_data is None:
            raise RuntimeError(f"LTRAnalysis '{self.analysis_identifier}': call fit() before compute_pdp().")

        pdp = compute_partial_dependence(
            model=self.ltr_model.booster,
            data=self.test_data,
            feature_cols=self.feature_cols,
            ranking_group_col=self.schema.ranking_group_col,
            tuner_col=self.schema.tuner_col,
            partition_name=self.analysis_identifier,
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
        if self.ltr_model is None or self.test_data is None:
            raise RuntimeError(f"LTRAnalysis '{self.analysis_identifier}': call fit() before compute_shap().")

        return run_shap_analysis(
            model=self.ltr_model.booster,
            data=self.test_data,
            feature_cols=self.feature_cols,
            output_dir=output_dir,
            top_k=top_k,
            sample_size=sample_size,
        )

    def compute_downsampling(
        self,
        sample_sizes: list[int],
        output_dir: Path | None = None,
    ) -> DownsamplingResults:
        if self.train_data is None or self.val_data is None or self.test_data is None or self.ltr_model is None:
            raise RuntimeError(f"LTRAnalysis '{self.analysis_identifier}': call fit() before compute_downsampling().")

        train_val = pd.concat([self.train_data, self.val_data], ignore_index=True)
        n_total = train_val[self.schema.ranking_group_col].nunique()
        val_prop = self.config.val_size / (self.config.train_size + self.config.val_size)

        sample_sizes = sorted({s for s in sample_sizes if s <= n_total} | {n_total})

        logger.info(f"Downsampling: {len(sample_sizes)} checkpoints, max {n_total} groups")

        all_groups = train_val[self.schema.ranking_group_col].unique()
        rng = np.random.RandomState(self.config.random_state)
        rows = []

        for n_groups in sample_sizes:
            groups = rng.choice(all_groups, size=n_groups, replace=False) if n_groups < n_total else all_groups
            subset = train_val[train_val[self.schema.ranking_group_col].isin(groups)].copy()

            n_val_groups = max(1, int(n_groups * val_prop))
            val_groups = rng.permutation(groups)[n_groups - n_val_groups:]
            sub_train = subset[~subset[self.schema.ranking_group_col].isin(val_groups)].copy()

            model = LTRModel(self.config.xgb_params, self.config.num_boost_rounds).fit(
                sub_train, self.schema, self.feature_cols,
            )
            metrics = _evaluate_rankings(
                self.test_data, model.predict(self.test_data),
                self.config.k_values, self.schema, ascending_scores=False,
            )
            rows.append({'sample_sizes': n_groups, 'n_train_groups': n_groups - n_val_groups,
                         'n_val_groups': n_val_groups, **metrics})

        df = pd.DataFrame(rows)
        metrics_dict = {col: df[col].tolist() for col in df.columns
                        if col.startswith('precision@') or col.startswith('ndcg@')}
        ds_results = DownsamplingResults(
            sample_sizes=df['sample_sizes'].tolist(),
            n_train_groups=df['n_train_groups'].tolist(),
            n_val_groups=df['n_val_groups'].tolist(),
            metrics=metrics_dict,
        )

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            df_dict = {
                'sample_sizes': ds_results.sample_sizes,
                'n_train_groups': ds_results.n_train_groups,
                'n_val_groups': ds_results.n_val_groups,
            }
            df_dict.update(ds_results.metrics)
            pd.DataFrame(df_dict).to_csv(output_dir / 'downsampling_curve.csv', index=False)
            plot_downsampling_curve(ds_results, output_dir / 'downsampling_curve.png', self.analysis_identifier)

        return ds_results

    def save(self, output_dir: Path) -> None:
        if self.test_data is None:
            raise RuntimeError(f"LTRAnalysis '{self.analysis_identifier}': call fit() before save().")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.test_data.to_csv(output_dir / 'test_data.csv', index=False)
        payload: dict = {
            'n_train': len(self.train_data) if self.train_data is not None else 0,
            'n_val': len(self.val_data) if self.val_data is not None else 0,
            'n_test': len(self.test_data),
        }
        if self.ltr_metrics:
            payload['ltr_metrics'] = self.ltr_metrics
            payload['naive_metrics'] = self.naive_metrics
        with open(output_dir / 'metrics.json', 'w') as fh:
            json.dump(payload, fh, indent=2)

    def summary(self) -> dict:
        if not self.ltr_metrics:
            raise RuntimeError(f"LTRAnalysis '{self.analysis_identifier}': call evaluate() before summary().")

        row = {
            'config': self.analysis_identifier,
            'partition': self.partition,
            'strategy': self.strategy,
            'n_train': len(self.train_data) if self.train_data is not None else 0,
            'n_val': len(self.val_data) if self.val_data is not None else 0,
            'n_test': len(self.test_data) if self.test_data is not None else 0,
        }
        for k, v in self.ltr_metrics.items():
            row[f'{k}_ltr'] = v
        for k, v in self.naive_metrics.items():
            row[f'{k}_naive'] = v
        return row
