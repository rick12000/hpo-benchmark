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
        self.baseline_ranker: AverageRankRanker | None = None
        self.ltr_metrics: dict[str, float] = {}
        self.naive_metrics: dict[str, float] = {}

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


    def fit(self, raw_data: pd.DataFrame, k_values: tuple[int, ...]) -> None:
        """Fit LTR and naive models, with optional hyperparameter tuning.

        Args:
            raw_data: Raw benchmark data.
            k_values: K values for evaluation metrics.
        """
        self._validate_strategy_eligibility(raw_data)

        data, self.feature_cols = prepare_data(
            raw_data=raw_data,
            partition=self.partition,
            schema=self.schema,
            metafeatures_schema=self.metafeatures_schema,
            synthetic_benchmark_id=self.synthetic_benchmark_id,
            tuner_encoding_method=self.tuner_encoding_method,
        )
        self.train_data, self.val_data, self.test_data = split_data(
            data=data,
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
            pd.concat([self.train_data, self.val_data], ignore_index=True),
            tuner_col=self.schema.tuner_col,
            label_col=self.schema.label_col,
        )

        logger.info(f"[{self.analysis_identifier}] fit – train={len(self.train_data)}, val={len(self.val_data)}, test={len(self.test_data)}")

    def evaluate(self, k_values: tuple[int, ...], output_dir: Path | None = None) -> dict:
        """Evaluate LTR and naive models on test data and optionally save results to disk.

        Args:
            k_values: K values for evaluation metrics.
            output_dir: If provided, metrics are written to ``metrics.json``.

        Returns:
            ``{'ltr_metrics': ..., 'naive_metrics': ..., 'n_test': int}``
        """
        if self.ltr_model is None or self.baseline_ranker is None or self.test_data is None:
            raise RuntimeError(f"LTRAnalysis '{self.analysis_identifier}': call fit() before evaluate().")

        def _rank_metrics(model, ascending: bool) -> dict[str, float]:
            return _evaluate_rankings(
                self.test_data,
                model.predict(self.test_data),
                k_values,
                self.schema.ranking_group_id_col,
                self.schema.label_col,
                self.schema.tuner_col,
                ascending_scores=ascending,
            )

        self.ltr_metrics = _rank_metrics(self.ltr_model, ascending=False)
        self.naive_metrics = _rank_metrics(self.baseline_ranker, ascending=True)

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

        return {'ltr_metrics': self.ltr_metrics, 'naive_metrics': self.naive_metrics, 'n_test': len(self.test_data)}


    def compute_pdp(
        self,
        output_dir: Path | None = None,
        n_grid_points: int = 20,
        show_ci: bool = True,
        n_bootstrap: int = 500,
        bootstrap_ci: float = 0.95,
    ) -> PartialDependenceResults:
        """Compute rank-based partial dependence plots on the test set.

        Sweeps each meta-feature across a grid while holding all other features
        at their observed values. Because meta-features are dataset-level quantities,
        the sweep is applied simultaneously to all tuners within every ranking group,
        preserving the competitive structure of the benchmark.

        Uncertainty is estimated by bootstrapping over ranking groups.

        Args:
            output_dir: If provided, PDP grid plots are saved here as PNG files.
            n_grid_points: Number of grid points for continuous features.
            show_ci: Whether to draw bootstrapped confidence interval bands on plots.
            n_bootstrap: Number of bootstrap resamples for CI estimation.
            bootstrap_ci: Coverage of the bootstrap confidence interval.

        Returns:
            ``PartialDependenceResults`` containing one result per (feature, tuner) pair.
        """
        if self.ltr_model is None or self.test_data is None:
            raise RuntimeError(f"LTRAnalysis '{self.analysis_identifier}': call fit() before compute_pdp().")

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
            plot_partial_dependence(pdp_results, Path(output_dir), show_ci=show_ci)
        return pdp_results

    def compute_shap(
        self,
        output_dir: Path | None = None,
        top_k: int = 20,
        sample_size: int | None = None,
    ) -> dict:
        """Compute rank-based SHAP values on the test set and optionally save plots and CSV.

        Args:
            output_dir: If provided, importance bar chart, beeswarm plot, and CSV are saved here.
            top_k: Number of top features to include in plots.
            sample_size: ShaRP perturbation sample size (``None`` → ShaRP default).

        Returns:
            ``{'shap_results': SharpResults, 'summary': pd.DataFrame}``
        """
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
        """Train LTR models at progressively smaller training set sizes and evaluate on the test set.

        Delegates to ``compute_downsampling_curve`` in the scaling module.

        Args:
            sample_sizes: Candidate group counts to evaluate.
            output_dir: If provided, results are written to CSV and a scaling curve plot is saved.

        Returns:
            ``DownsamplingResults`` containing metric trajectories across checkpoints.
        """
        if self.train_data is None or self.val_data is None or self.test_data is None or self.ltr_model is None:
            raise RuntimeError(f"LTRAnalysis '{self.analysis_identifier}': call fit() before compute_downsampling().")

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
            plot_downsampling_curve(downsampling_results, output_dir / 'downsampling_curve.png', self.analysis_identifier)

        return downsampling_results

