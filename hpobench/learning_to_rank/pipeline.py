"""
LTRPipeline: runs one LTRAnalysis per (partition, strategy) pair and exposes
aggregated training, evaluation, explainability, and persistence as methods.

Typical usage::

    pipeline = LTRPipeline(config=LTRConfig(), schema=BenchmarkDataSchema())
    pipeline.fit(raw_data).evaluate()

    pipeline.compute_pdp(output_dir=Path('output/pdp'))
    pipeline.compute_downsampling(output_dir=Path('output/ds'))
    pipeline.save(Path('output'))

    print(pipeline.summary())
    analysis = pipeline['all_random']
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Literal, NamedTuple

from hpobench.config.schema import BenchmarkDataSchema
from hpobench.learning_to_rank.model import LTRConfig
from hpobench.learning_to_rank.analysis import LTRAnalysis

logger = logging.getLogger(__name__)


class AnalysisConfig(NamedTuple):
    """Declarative specification for one LTR analysis run."""
    name: str
    partition: Literal['all', 'synthetic', 'real']
    strategy: Literal['random', 'synthetic_train_real_test']


ANALYSIS_CONFIGS: list[AnalysisConfig] = [
    AnalysisConfig('all_random',                     'all',       'random'),
    AnalysisConfig('all_synthetic_train_real_test',  'all',       'synthetic_train_real_test'),
    AnalysisConfig('synthetic_random',               'synthetic', 'random'),
    AnalysisConfig('real_random',                    'real',      'random'),
]


class LTRPipeline:
    """Orchestrates LTR analyses across multiple data partitions."""

    def __init__(
        self,
        config: LTRConfig | None = None,
        schema: BenchmarkDataSchema | None = None,
        analysis_configs: list[AnalysisConfig] | None = None,
        tuner_encoding_method: Literal['ordinal', 'one_hot'] = 'ordinal',
    ) -> None:
        self.config = config or LTRConfig()
        self.schema = schema or BenchmarkDataSchema()
        self.analysis_configs = analysis_configs or ANALYSIS_CONFIGS
        self.tuner_encoding_method = tuner_encoding_method
        self.analyses: dict[str, LTRAnalysis] = {}

    def __getitem__(self, name: str) -> LTRAnalysis:
        return self.analyses[name]

    def __len__(self) -> int:
        return len(self.analyses)

    def fit(self, raw_data: pd.DataFrame) -> "LTRPipeline":
        """Train models for all configurations, skipping any that lack sufficient data."""
        for ac in self.analysis_configs:
            analysis = LTRAnalysis(
                config=self.config,
                schema=self.schema,
                partition=ac.partition,
                strategy=ac.strategy,
                tuner_encoding_method=self.tuner_encoding_method,
                name=ac.name,
            )
            try:
                analysis.fit(raw_data)
                self.analyses[ac.name] = analysis
            except ValueError as exc:
                logger.warning(f"Skipping '{ac.name}': {exc}")
        return self

    def evaluate(self) -> "LTRPipeline":
        """Compute test-set metrics for all fitted analyses."""
        for analysis in self.analyses.values():
            analysis.evaluate()
        return self

    def compute_pdp(self, output_dir: Path, n_grid_points: int = 20, show_std: bool = True) -> dict:
        """Compute rank-based PDPs for every fitted analysis."""
        results = {}
        for name, analysis in self.analyses.items():
            try:
                results[name] = analysis.compute_pdp(
                    output_dir=Path(output_dir) / name,
                    n_grid_points=n_grid_points,
                    show_std=show_std,
                )
            except Exception as exc:
                logger.warning(f"compute_pdp failed for '{name}': {exc}")
        return results

    def compute_downsampling(self, output_dir: Path, sample_sizes: list[int] | None = None) -> dict:
        """Compute scaling curves for every fitted analysis."""
        results = {}
        for name, analysis in self.analyses.items():
            try:
                results[name] = analysis.compute_downsampling(
                    output_dir=Path(output_dir) / name,
                    sample_sizes=sample_sizes,
                )
            except Exception as exc:
                logger.warning(f"compute_downsampling failed for '{name}': {exc}")
        return results

    def save(self, output_dir: Path) -> None:
        """Write summary.csv and per-partition test_data.csv / metrics.json."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.summary().to_csv(output_dir / 'summary.csv', index=False)
        for name, analysis in self.analyses.items():
            analysis.save(output_dir / name)

    def summary(self) -> pd.DataFrame:
        """DataFrame with one row per fitted and evaluated analysis."""
        return pd.DataFrame([a.summary() for a in self.analyses.values()])
