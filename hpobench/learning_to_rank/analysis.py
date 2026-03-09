import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from hpobench.config.schema import BenchmarkDataSchema, SurrogateMetafeaturesSchema
from hpobench.config.types import (
    LTRConfig,
    LTRTuningConfig,
    LTRHyperparameters,
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
    ranking_group_col: str,
    label_col: str,
    tuner_col: str,
    ascending_scores: bool = False,
) -> dict[str, float]:
    """Compute precision@k and NDCG@k across all ranking groups."""
    from sklearn.metrics import ndcg_score

    test_data = test_data.copy()
    test_data['_pred_score'] = predicted_scores

    precision_lists = {k: [] for k in k_values}
    ndcg_lists = {k: [] for k in k_values}

    for _, group in test_data.groupby(ranking_group_col):
        true_sorted = group.sort_values(label_col, ascending=False)
        pred_sorted = group.sort_values('_pred_score', ascending=ascending_scores)

        true_ranking = true_sorted[tuner_col].tolist()
        pred_ranking = pred_sorted[tuner_col].tolist()

        true_relevance = np.arange(len(true_sorted), 0, -1)
        relevance_map = dict(zip(true_sorted[tuner_col], true_relevance))
        pred_relevance = np.array([relevance_map[t] for t in pred_ranking])

        for k in k_values:
            precision_lists[k].append(_precision_at_k(pred_ranking, true_ranking, k))
            ndcg_lists[k].append(ndcg_score([true_relevance], [pred_relevance], k=k))

    results = {}
    for k in k_values:
        results[f'precision@{k}'] = float(np.mean(precision_lists[k]))
        results[f'ndcg@{k}'] = float(np.mean(ndcg_lists[k]))
    
    return results




class LTRAnalysis:
    """Single LTR experiment with mutable state for fit results."""

    def __init__(
        self,
        schema: BenchmarkDataSchema,
        metafeatures_schema: SurrogateMetafeaturesSchema,
        synthetic_benchmark_id: str,
        ltr_config: LTRConfig,
        partition: Partition = 'all',
        strategy: SplitStrategy = 'random',
        tuner_encoding_method: TunerEncoding = 'ordinal',
    ) -> None:
        self.config = ltr_config
        self.schema = schema
        self.metafeatures_schema = metafeatures_schema
        self.synthetic_benchmark_id = synthetic_benchmark_id
        self.partition = partition
        self.strategy = strategy
        self.tuner_encoding_method = tuner_encoding_method
        self.analysis_identifier = f"{partition}_{strategy}_{tuner_encoding_method}"

        self.train_data: pd.DataFrame | None = None
        self.val_data: pd.DataFrame | None = None
        self.test_data: pd.DataFrame | None = None
        self.feature_cols: list[str] = []
        self.ltr_model: LTRModel | None = None
        self.naive_ranker: NaiveRanker | None = None
        self.ltr_metrics: dict[str, float] = {}
        self.naive_metrics: dict[str, float] = {}
        self.best_hyperparameters: LTRHyperparameters | None = None

    def _validate_strategy_eligibility(self, raw_data: pd.DataFrame) -> None:
        """Validate that the strategy is compatible with the data and partition."""
        if self.strategy == 'synthetic_train_real_test':
            if self.partition != 'all':
                raise ValueError(
                    f"Strategy 'synthetic_train_real_test' requires partition='all' "
                    f"but got partition='{self.partition}' for config '{self.analysis_identifier}'"
                )
            
            has_synthetic = (raw_data[self.schema.benchmark_identifier_col] == self.synthetic_benchmark_id).any()
            has_real = (raw_data[self.schema.benchmark_identifier_col] != self.synthetic_benchmark_id).any()
            
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

    def _generate_hyperparameter_candidates(
        self,
        n_candidates: int,
        tuning_config: LTRTuningConfig,
    ) -> list[LTRHyperparameters]:
        """Generate random hyperparameter configurations from the search space.
        
        Args:
            n_candidates: Number of hyperparameter configurations to generate.
            tuning_config: Tuning configuration containing search space.
            
        Returns:
            List of hyperparameter configurations.
        """
        rng = np.random.RandomState(tuning_config.tuning_random_state)
        search_space = tuning_config.search_space
        
        candidates = []
        for _ in range(n_candidates):
            hyperparams = LTRHyperparameters(
                num_boost_rounds=int(rng.choice(search_space.num_boost_rounds)),
                learning_rate=float(rng.choice(search_space.learning_rate)),
                max_depth=int(rng.choice(search_space.max_depth)),
                subsample=float(rng.choice(search_space.subsample)),
                colsample_bytree=float(rng.choice(search_space.colsample_bytree)),
                objective=tuning_config.default_hyperparameters.objective,
                verbosity=tuning_config.default_hyperparameters.verbosity,
                seed=tuning_config.default_hyperparameters.seed,
            )
            candidates.append(hyperparams)
        
        return candidates

    def _train_ltr_model(
        self,
        train_data: pd.DataFrame,
        hyperparameters: LTRHyperparameters,
    ) -> LTRModel:
        """Train an LTR model with given hyperparameters.
        
        Args:
            train_data: Training data.
            hyperparameters: Model hyperparameters.
            
        Returns:
            Trained LTR model.
        """
        model = LTRModel(
            num_boost_rounds=hyperparameters.num_boost_rounds,
            objective=hyperparameters.objective,
            learning_rate=hyperparameters.learning_rate,
            max_depth=hyperparameters.max_depth,
            subsample=hyperparameters.subsample,
            colsample_bytree=hyperparameters.colsample_bytree,
            verbosity=hyperparameters.verbosity,
            seed=hyperparameters.seed,
        )
        model.fit(
            train_data,
            group_col=self.schema.ranking_group_col,
            label_col=self.schema.label_col,
            feature_cols=self.feature_cols,
        )
        return model

    def _evaluate_single_metric(
        self,
        model: LTRModel,
        val_data: pd.DataFrame,
        metric_name: str,
        k_values: tuple[int, ...],
    ) -> float:
        """Evaluate a single metric for hyperparameter tuning.
        
        Args:
            model: Trained LTR model.
            val_data: Validation data.
            metric_name: Name of the metric (e.g., 'precision@1').
            k_values: K values for evaluation.
            
        Returns:
            Metric value.
        """
        metrics = _evaluate_rankings(
            val_data,
            model.predict(val_data),
            k_values,
            self.schema.ranking_group_col,
            self.schema.label_col,
            self.schema.tuner_col,
            ascending_scores=False,
        )
        return metrics[metric_name]

    def _tune_hyperparameters(
        self,
        tuning_config: LTRTuningConfig,
        k_values: tuple[int, ...],
    ) -> LTRHyperparameters:
        """Tune LTR model hyperparameters using random search.
        
        Args:
            tuning_config: Configuration for hyperparameter tuning.
            k_values: K values for evaluation metrics.
            
        Returns:
            Best hyperparameters found.
        """
        if self.train_data is None or self.val_data is None:
            raise RuntimeError("train_data and val_data must be set before tuning")

        logger.info(
            f"[{self.analysis_identifier}] Starting hyperparameter tuning with "
            f"{tuning_config.n_tuning_trials} trials, optimizing {tuning_config.tuning_metric}"
        )

        candidates = self._generate_hyperparameter_candidates(
            n_candidates=tuning_config.n_tuning_trials,
            tuning_config=tuning_config,
        )

        best_score = -np.inf
        best_hyperparams = tuning_config.default_hyperparameters

        for idx, hyperparams in enumerate(candidates, 1):
            model = self._train_ltr_model(self.train_data, hyperparams)
            score = self._evaluate_single_metric(
                model,
                self.val_data,
                tuning_config.tuning_metric,
                k_values,
            )
            
            logger.info(
                f"[{self.analysis_identifier}] Trial {idx}/{tuning_config.n_tuning_trials}: "
                f"{tuning_config.tuning_metric}={score:.4f}"
            )
            
            if score > best_score:
                best_score = score
                best_hyperparams = hyperparams

        logger.info(
            f"[{self.analysis_identifier}] Best {tuning_config.tuning_metric}: {best_score:.4f}"
        )

        return best_hyperparams

    def fit(self, raw_data: pd.DataFrame, k_values: tuple[int, ...]) -> None:
        """Fit LTR and naive models with optional hyperparameter tuning.
        
        Args:
            raw_data: Raw benchmark data.
            k_values: K values for evaluation metrics.
        """
        self._validate_strategy_eligibility(raw_data)

        data, feature_cols = prepare_data(
            raw_data=raw_data,
            partition=self.partition,
            schema=self.schema,
            metafeatures_schema=self.metafeatures_schema,
            synthetic_benchmark_id=self.synthetic_benchmark_id,
            tuner_encoding_method=self.tuner_encoding_method,
        )
        self.feature_cols = feature_cols
        
        self.train_data, self.val_data, self.test_data = split_data(
            data=data,
            strategy=self.strategy,
            train_size=self.config.train_size,
            val_size=self.config.val_size,
            random_state=self.config.random_state,
            synthetic_benchmark_id=self.synthetic_benchmark_id,
            schema=self.schema,
        )

        tuning_config = self.config.tuning
        
        if tuning_config.n_tuning_trials > 1:
            self.best_hyperparameters = self._tune_hyperparameters(tuning_config, k_values)
            train_val_data = pd.concat([self.train_data, self.val_data], ignore_index=True)
            self.ltr_model = self._train_ltr_model(train_val_data, self.best_hyperparameters)
        else:
            self.best_hyperparameters = tuning_config.default_hyperparameters
            train_val_data = pd.concat([self.train_data, self.val_data], ignore_index=True)
            self.ltr_model = self._train_ltr_model(train_val_data, self.best_hyperparameters)

        self.naive_ranker = NaiveRanker()
        train_val_data = pd.concat([self.train_data, self.val_data], ignore_index=True)
        self.naive_ranker.fit(
            train_val_data,
            tuner_col=self.schema.tuner_col,
            label_col=self.schema.label_col,
        )

        logger.info(
            f"[{self.analysis_identifier}] fit – train={len(self.train_data)}, "
            f"val={len(self.val_data)}, test={len(self.test_data)}"
        )

    def evaluate(self, k_values: tuple[int, ...]) -> None:
        """Evaluate LTR and naive models on test data.
        
        Args:
            k_values: K values for evaluation metrics.
        """
        if self.ltr_model is None or self.naive_ranker is None or self.test_data is None:
            raise RuntimeError(f"LTRAnalysis '{self.analysis_identifier}': call fit() before evaluate().")

        model_configs = {
            'ltr': {'model': self.ltr_model, 'ascending_scores': False},
            'naive': {'model': self.naive_ranker, 'ascending_scores': True},
        }
        
        for model_type, config in model_configs.items():
            metrics = _evaluate_rankings(
                self.test_data,
                config['model'].predict(self.test_data),
                k_values,
                self.schema.ranking_group_col,
                self.schema.label_col,
                self.schema.tuner_col,
                ascending_scores=config['ascending_scores'],
            )
            if model_type == 'ltr':
                self.ltr_metrics = metrics
            elif model_type == 'naive':
                self.naive_metrics = metrics


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
        """Compute downsampling curve using best hyperparameters.
        
        Args:
            sample_sizes: List of sample sizes to evaluate.
            output_dir: Optional output directory for saving results.
            
        Returns:
            Downsampling results.
        """
        if self.train_data is None or self.val_data is None or self.test_data is None or self.ltr_model is None:
            raise RuntimeError(f"LTRAnalysis '{self.analysis_identifier}': call fit() before compute_downsampling().")

        if self.best_hyperparameters is None:
            raise RuntimeError(f"LTRAnalysis '{self.analysis_identifier}': best_hyperparameters not set.")

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

            model = self._train_ltr_model(sub_train, self.best_hyperparameters)
            
            metrics = _evaluate_rankings(
                self.test_data,
                model.predict(self.test_data),
                self.config.k_values,
                self.schema.ranking_group_col,
                self.schema.label_col,
                self.schema.tuner_col,
                ascending_scores=False,
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
        
        payload = {
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
