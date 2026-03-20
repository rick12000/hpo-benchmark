import json
import logging
import pandas as pd
from pathlib import Path

from hpobench.config.schema import BenchmarkDataSchema, SurrogateMetafeaturesSchema
from hpobench.config.types import (
    LTRConfig,
    Partition,
    SplitStrategy,
    TunerEncoding,
    PartialDependenceResults,
    DownsamplingResults,
)
from hpobench.learning_to_rank.model import AverageRankRanker, LTRModel, _evaluate_rankings
from hpobench.learning_to_rank.preprocessing import prepare_data, split_data
from hpobench.learning_to_rank.explainability import (
    compute_partial_dependence,
    plot_partial_dependence,
    run_shap_analysis,
)
from hpobench.learning_to_rank.scaling import compute_downsampling_curve, plot_downsampling_curve

logger = logging.getLogger(__name__)


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
        """Initialize LTR analysis with configuration.
        
        Args:
            schema: Column name schema.
            metafeatures_schema: Feature column schema.
            synthetic_benchmark_id: Synthetic benchmark identifier.
            ltr_config: LTR configuration.
            partition: Data partition. Defaults to 'all'.
            strategy: Split strategy. Defaults to 'random'.
            tuner_encoding_method: Tuner encoding method. Defaults to 'ordinal'.
        """
        self.config = ltr_config
        self.schema = schema
        self.metafeatures_schema = metafeatures_schema
        self.synthetic_benchmark_id = synthetic_benchmark_id
        self.partition = partition
        self.strategy = strategy
        self.tuner_encoding_method = tuner_encoding_method
        self.analysis_identifier = f'{partition}_{strategy}_{tuner_encoding_method}'

        self.train_data: pd.DataFrame | None = None
        self.val_data: pd.DataFrame | None = None
        self.test_data: pd.DataFrame | None = None
        self.feature_cols: list[str] = []
        self.ltr_model: LTRModel | None = None
        self.baseline_ranker: AverageRankRanker | None = None
        self.ltr_metrics: dict[str, float] = {}
        self.naive_metrics: dict[str, float] = {}

    def _validate_strategy_eligibility(self, raw_data: pd.DataFrame) -> None:
        """Validate strategy compatibility with data and partition."""
        if self.strategy == 'synthetic_train_real_test':
            if self.partition != 'all':
                raise ValueError(
                    f'synthetic_train_real_test requires partition=all, got {self.partition!r} '
                    f'for {self.analysis_identifier}'
                )

            has_synthetic = (raw_data[self.schema.benchmark_identifier_col] == self.synthetic_benchmark_id).any()
            has_real = (raw_data[self.schema.benchmark_identifier_col] != self.synthetic_benchmark_id).any()

            if not has_synthetic:
                raise ValueError(
                    f'synthetic_train_real_test requires synthetic data '
                    f'but none found in {self.analysis_identifier}'
                )
            if not has_real:
                raise ValueError(
                    f'synthetic_train_real_test requires real data for testing '
                    f'but none found in {self.analysis_identifier}'
                )


    def fit(self, raw_data: pd.DataFrame, k_values: tuple[int, ...]) -> None:
        """Fit LTR and baseline models with optional hyperparameter tuning.

        Args:
            raw_data: Raw benchmark data.
            k_values: K values for evaluation metrics.
        """
        self._validate_strategy_eligibility(raw_data)

        if self.partition == 'synthetic':
            data_to_process = raw_data[raw_data[self.schema.benchmark_identifier_col] == self.synthetic_benchmark_id]
        elif self.partition == 'real':
            data_to_process = raw_data[raw_data[self.schema.benchmark_identifier_col] != self.synthetic_benchmark_id]
        else:
            data_to_process = raw_data

        if data_to_process.empty:
            raise ValueError(f"No data available for partition {self.partition!r}")

        data, self.feature_cols = prepare_data(
            raw_data=data_to_process,
            schema=self.schema,
            metafeatures_schema=self.metafeatures_schema,
            tuner_encoding_method=self.tuner_encoding_method,
        )
        self.train_data, self.val_data, self.test_data = split_data(
            data=data,
            partition=self.partition,
            strategy=self.strategy,
            train_size=self.config.train_size,
            val_size=self.config.val_size,
            random_state=self.config.random_state,
            synthetic_benchmark_id=self.synthetic_benchmark_id,
            schema=self.schema,
        )

        self.ltr_model = LTRModel()
        self.ltr_model.fit(
            train_data=self.train_data,
            val_data=self.val_data,
            group_col=self.schema.ranking_group_id_col,
            label_col=self.schema.label_col,
            tuner_col=self.schema.tuner_col,
            feature_cols=self.feature_cols,
            tuning_config=self.config.tuning,
            k_values=k_values,
            analysis_identifier=self.analysis_identifier,
            n_tuning_trials=20,
        )

        self.baseline_ranker = AverageRankRanker()
        self.baseline_ranker.fit(
            train_data=pd.concat([self.train_data, self.val_data], ignore_index=True),
            tuner_col=self.schema.tuner_col,
            label_col=self.schema.label_col,
        )

        logger.info(
            f'[{self.analysis_identifier}] fit – train={len(self.train_data)}, '
            f'val={len(self.val_data)}, test={len(self.test_data)}'
        )

    def evaluate(self, k_values: tuple[int, ...], output_dir: Path | None = None) -> dict:
        """Evaluate models on test data and optionally save results.

        Args:
            k_values: K values for evaluation metrics.
            output_dir: If provided, metrics written to metrics.json.

        Returns:
            Dict with 'ltr_metrics', 'naive_metrics', and 'n_test'.
        """
        if self.ltr_model is None or self.baseline_ranker is None or self.test_data is None:
            raise RuntimeError(
                f'LTRAnalysis {self.analysis_identifier!r}: call fit() before evaluate().'
            )

        def _rank_metrics(model) -> dict[str, float]:
            return _evaluate_rankings(
                test_data=self.test_data,
                predicted_scores=model.predict(self.test_data),
                k_values=k_values,
                ranking_group_id_col=self.schema.ranking_group_id_col,
                label_col=self.schema.label_col,
                tuner_col=self.schema.tuner_col,
            )

        self.ltr_metrics = _rank_metrics(self.ltr_model)
        self.naive_metrics = _rank_metrics(self.baseline_ranker)

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                'n_train': len(self.train_data) if self.train_data is not None else 0,
                'n_val': len(self.val_data) if self.val_data is not None else 0,
                'n_test': len(self.test_data),
                'ltr_metrics': self.ltr_metrics,
                'naive_metrics': self.naive_metrics,
            }
            with open(output_dir / 'metrics.json', 'w') as f:
                json.dump(payload, f, indent=2)

        return {
            'ltr_metrics': self.ltr_metrics,
            'naive_metrics': self.naive_metrics,
            'n_test': len(self.test_data),
        }


    def compute_pdp(
        self,
        output_dir: Path | None = None,
        n_grid_points: int = 20,
        show_ci: bool = True,
        n_bootstrap: int = 500,
        bootstrap_ci: float = 0.95,
    ) -> PartialDependenceResults:
        """Compute rank-based partial dependence plots on test set.

        Args:
            output_dir: Directory to save PDP plots.
            n_grid_points: Grid points for continuous features.
            show_ci: Whether to show bootstrap confidence intervals.
            n_bootstrap: Bootstrap resamples for CI.
            bootstrap_ci: CI coverage.

        Returns:
            PartialDependenceResults with one result per (feature, tuner) pair.
        """
        if self.ltr_model is None or self.test_data is None:
            raise RuntimeError(
                f'LTRAnalysis {self.analysis_identifier!r}: call fit() before compute_pdp().'
            )

        pdp_results = compute_partial_dependence(
            model=self.ltr_model.booster,
            data=self.test_data,
            feature_cols=self.feature_cols,
            group_col=self.schema.ranking_group_id_col,
            tuner_col=self.schema.tuner_col,
            partition_name=self.analysis_identifier,
            n_grid_points=n_grid_points,
            n_bootstrap=n_bootstrap,
            bootstrap_ci=bootstrap_ci,
        )
        if output_dir is not None:
            plot_partial_dependence(pdp_results=pdp_results, output_dir=Path(output_dir), show_ci=show_ci)
        return pdp_results

    def compute_shap(
        self,
        output_dir: Path | None = None,
        top_k: int = 20,
        sample_size: int = 20,
        n_groups: int | None = None,
        verbose: int = 1,
    ) -> dict:
        """Compute rank-dependent SHAP values via ShaRP and save plots and summary.

        Args:
            output_dir: Directory to save plots and CSV.
            top_k: Top features to include in plots.
            sample_size: Coalitions sampled per data point by ShaRP.
            n_groups: If set, subsample this many complete ranking groups before
                computing SHAP. Group-level subsampling preserves rank coupling.
            verbose: Verbosity passed to ShaRP (0 = silent, 1 = progress bar).

        Returns:
            Dict with 'shap_results' and 'summary'.
        """
        if self.ltr_model is None or self.test_data is None:
            raise RuntimeError(
                f'LTRAnalysis {self.analysis_identifier!r}: call fit() before compute_shap().'
            )

        return run_shap_analysis(
            model=self.ltr_model.booster,
            data=self.test_data,
            feature_cols=self.feature_cols,
            group_col=self.schema.ranking_group_id_col,
            output_dir=output_dir,
            top_k=top_k,
            sample_size=sample_size,
            n_groups=n_groups,
            verbose=verbose,
        )

    def compute_downsampling(
        self,
        sample_sizes: list[int],
        output_dir: Path | None = None,
    ) -> DownsamplingResults:
        """Train models at progressively smaller sizes and evaluate on test set.

        Args:
            sample_sizes: Group counts to evaluate.
            output_dir: Directory to save CSV and plot.

        Returns:
            DownsamplingResults with metric trajectories.
        """
        if self.train_data is None or self.val_data is None or self.test_data is None or self.ltr_model is None:
            raise RuntimeError(
                f'LTRAnalysis {self.analysis_identifier!r}: call fit() before compute_downsampling().'
            )

        train_val_data = pd.concat([self.train_data, self.val_data], ignore_index=True)
        downsampling_results = compute_downsampling_curve(
            train_val_data=train_val_data,
            test_data=self.test_data,
            feature_cols=self.feature_cols,
            requested_sample_sizes=sample_sizes,
            ltr_config=self.config,
            schema=self.schema,
            analysis_identifier=self.analysis_identifier,
        )

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({
                'sample_sizes': downsampling_results.sample_sizes,
                'n_train_groups': downsampling_results.n_train_groups,
                'n_val_groups': downsampling_results.n_val_groups,
                **downsampling_results.metrics,
            }).to_csv(output_dir / 'downsampling_curve.csv', index=False)
            plot_downsampling_curve(
                downsampling_results=downsampling_results,
                output_path=output_dir / 'downsampling_curve.png',
                partition_name=self.analysis_identifier,
            )

        return downsampling_results

