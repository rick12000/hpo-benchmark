import gc
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

try:
    from confopt.selection.conformalization import QuantileConformalEstimator
    from confopt.utils.configurations.encoding import ConfigurationEncoder
except ImportError:
    raise ImportError(
        "confopt is a core dependency of this repository, but it is not automatically installed via pyproject.toml, please refer to the README.md for instructions on how to install this separately"
    )

from hpobench.config.constants import ExperimentParameters, SyntheticGenerationParameters
from hpobench.config.schema import BenchmarkDataSchema, Aliases, SurrogateMetafeaturesSchema
from hpobench.config.types import AnalysisConfig, CategoricalRange, CustomGPModel, ExperimentConfig, FloatRange, IntRange, LTRConfig, TunerConfig, TunerEncoding, WarmStartStrategy
from hpobench.learning_to_rank.analysis import LTRAnalysis
from hpobench.orchestration.prepare import setup_synthetic_configs, setup_yahpo_instance_configs
from hpobench.tuning.tune import tune
from hpobench.utils import add_runtime, generate_hyperparameter_combinations

logger = logging.getLogger(__name__)
os.environ["SYNETUNE_FOLDER"] = "cache/syne-tune"

aliases = Aliases()


def load_experiment_configs(
    benchmarks: list[
        Literal[
            "lcbench",
            "rbv2_aknn",
            "LCBench-L",
            "LCBench-H",
            "LCBench-A",
            "rbv2_aknn-L",
            "rbv2_aknn-H",
            "rbv2_aknn-A",
            "synthetic_tabular",
        ]
    ],
    tuning_configurations: list[TunerConfig],
    max_n_instances_per_benchmark: int = 10,
    datasets_per_benchmark: Optional[list[list[str]]] = None,
    synthetic_tabular_ids: Optional[list[str]] = None,
) -> list[ExperimentConfig]:
    """Load and configure benchmark instances for HPO experiments.

    Args:
        benchmarks: Benchmark names to initialize. Supports YAHPO variants and synthetic_tabular.
        tuning_configurations: Tuner configurations defining the HPO algorithms to compare.
        max_n_instances_per_benchmark: Maximum dataset instances per benchmark.
        datasets_per_benchmark: Per-benchmark list of dataset identifiers; overrides sampling.
        synthetic_tabular_ids: Dataset IDs available for the synthetic_tabular benchmark.

    Returns:
        List of ExperimentConfig objects pairing each dataset instance with its search space,
        objective function, and tuning parameters.
    """
    logger.info("Setting up benchmark instances...")

    experiment_configs: list[ExperimentConfig] = []
    for i, benchmark in enumerate(benchmarks):
        if benchmark in [
            "rbv2_aknn",
            "lcbench",
            "LCBench-L",
            "LCBench-H",
            "LCBench-A",
            "rbv2_aknn-L",
            "rbv2_aknn-H",
            "rbv2_aknn-A",
        ]:
            configs = setup_yahpo_instance_configs(
                benchmark=benchmark,
                tuning_configurations=tuning_configurations,
                max_n_instances=max_n_instances_per_benchmark,
            )
            experiment_configs.extend(configs)

    if "synthetic_tabular" in benchmarks:
        idx = benchmarks.index("synthetic_tabular")
        all_datasets = synthetic_tabular_ids if synthetic_tabular_ids is not None else []
        if datasets_per_benchmark is not None and datasets_per_benchmark[idx] is not None:
            selected_datasets = datasets_per_benchmark[idx]
        elif max_n_instances_per_benchmark < len(all_datasets):
            selected_datasets = all_datasets[:max_n_instances_per_benchmark]
        else:
            selected_datasets = all_datasets

        configs = setup_synthetic_configs(
            datasets=selected_datasets,
            tuning_configurations=tuning_configurations,
        )
        experiment_configs.extend(configs)

    return experiment_configs


def _generate_random_warm_starts(
    search_space: dict[str, FloatRange | IntRange | CategoricalRange],
    n_configs: int,
    random_state: int,
    objective_function: object,
) -> list[tuple[dict, float]]:
    """Sample random warm start configurations and evaluate them.

    Args:
        search_space: Hyperparameter search space definition.
        n_configs: Number of configurations to sample.
        random_state: Random seed for reproducibility.
        objective_function: Surrogate model with a ``predict_batch`` method.

    Returns:
        List of (configuration, performance) tuples.
    """
    configs = generate_hyperparameter_combinations(
        params=search_space,
        n_combinations=n_configs,
        random_state=random_state,
    )
    performances = objective_function.predict_batch(configs)
    return list(zip(configs, performances))


def _generate_gp_warm_starts(
    search_space: dict[str, FloatRange | IntRange | CategoricalRange],
    n_initial_random: int,
    n_gp_searches: int,
    random_state: int,
    objective_function: object,
    acquisition_strategy: Literal["TS", "EI"],
) -> list[tuple[dict, float]]:
    """Generate warm start configurations via GP-guided optimization.

    First draws ``n_initial_random`` random configs to seed the GP, then runs
    ``n_gp_searches`` acquisition-guided trials. Only the GP-searched configs
    are returned; the seeding randoms are discarded.

    Args:
        search_space: Hyperparameter search space definition.
        n_initial_random: Random configs used to seed the GP (not returned).
        n_gp_searches: Number of GP-guided acquisition trials to return.
        random_state: Random seed for reproducibility.
        objective_function: Surrogate model with a ``predict_batch`` method.
        acquisition_strategy: Acquisition function — ``"TS"`` (Thompson Sampling)
            or ``"EI"`` (Expected Improvement).

    Returns:
        List of (configuration, performance) tuples from GP-guided trials only.
    """
    initial_warm_starts = _generate_random_warm_starts(
        search_space=search_space,
        n_configs=n_initial_random,
        random_state=random_state,
        objective_function=objective_function,
    )

    gp_tuner_config = TunerConfig(
        tuner=CustomGPModel(backend="gp_opt", searcher=acquisition_strategy),
        tuner_identifier=f"GP-{acquisition_strategy}",
    )

    history = tune(
        performance_generator=objective_function,
        tuner_config=gp_tuner_config,
        params=search_space,
        warm_start_configs=initial_warm_starts,
        random_state=random_state,
        n_trials=n_initial_random + n_gp_searches,
        timeout=None,
    )

    gp_searched_history = history.tail(n_gp_searches)
    return [
        (row["configurations"], row["performance"])
        for _, row in gp_searched_history.iterrows()
    ]


def _annotate_trial_result(
    trial_row: pd.DataFrame,
    experiment_config: ExperimentConfig,
    tuner: TunerConfig,
    repetition: int,
    n_ws: int,
    strategy: WarmStartStrategy,
    surrogate_metafeatures: dict,
    aliases: Aliases,
) -> pd.DataFrame:
    """Attach experiment metadata and tuner-specific fields to a trial result row.

    Args:
        trial_row: Single-row DataFrame from ``tune()`` output to annotate.
        experiment_config: Config containing benchmark/dataset identifiers.
        tuner: Tuner configuration providing algorithm metadata.
        repetition: Zero-indexed repetition number (stored as 1-indexed).
        n_ws: Number of warm start configurations used.
        strategy: Warm start generation strategy applied.
        surrogate_metafeatures: Metafeature dict to merge into the row.
        aliases: Alias mappings for benchmark, architecture, and sampler names.

    Returns:
        Annotated copy of ``trial_row`` with all metadata columns populated.
    """
    trial_row = trial_row.copy()

    aliased_benchmark_identifier = (
        aliases.benchmark_aliases.get(experiment_config.benchmark_identifier)
        or experiment_config.benchmark_identifier
    )
    trial_row["benchmark_identifier"] = aliased_benchmark_identifier
    trial_row["dataset"] = experiment_config.dataset_identifier
    trial_row["tuner"] = tuner.tuner_identifier
    trial_row["repetition"] = repetition + 1
    trial_row["searcher_tuning_framework"] = tuner.searcher_tuning_framework
    trial_row["n_random_warm_starts"] = n_ws
    trial_row["warm_start_strategy"] = strategy

    for key, value in surrogate_metafeatures.items():
        trial_row[key] = value

    if tuner.tuner.backend == "confopt":
        sampler_name = tuner.tuner.searcher.sampler.__class__.__name__
        confidence_level = str(tuner.tuner.searcher.sampler.interval_width) if hasattr(tuner.tuner.searcher.sampler, "interval_width") else ""
        estimator_architecture = tuner.tuner.searcher.quantile_estimator_architecture
        n_pre_conformal_trials = tuner.tuner.searcher.n_pre_conformal_trials if hasattr(tuner.tuner.searcher, "n_pre_conformal_trials") else ""
        sampler_n_quantiles = tuner.tuner.searcher.sampler.n_quantiles if hasattr(tuner.tuner.searcher.sampler, "n_quantiles") else ""
        sampler_adapter = (
            "None" if tuner.tuner.searcher.sampler.adapter is None
            else str(tuner.tuner.searcher.sampler.adapter)
        ) if hasattr(tuner.tuner.searcher.sampler, "adapter") else ""
        tuner_searcher_tuning_framework = "None" if tuner.searcher_tuning_framework is None else str(tuner.searcher_tuning_framework)
    else:
        sampler_name = confidence_level = estimator_architecture = ""
        n_pre_conformal_trials = sampler_n_quantiles = sampler_adapter = ""
        tuner_searcher_tuning_framework = ""

    aliased_estimator_architecture = aliases.architecture_aliases.get(estimator_architecture) or estimator_architecture
    aliased_sampler_name = aliases.sampler_aliases.get(sampler_name) or sampler_name
    if tuner.tuner.backend == "confopt" and sampler_name == "ThompsonSampler":
        if tuner.tuner.searcher.sampler.enable_optimistic_sampling:
            aliased_sampler_name = "OBS"

    trial_row["estimator_architecture"] = aliased_estimator_architecture
    trial_row["confidence_level"] = confidence_level
    trial_row["sampler"] = aliased_sampler_name
    trial_row["n_pre_conformal_trials"] = n_pre_conformal_trials
    trial_row["sampler_n_quantiles"] = sampler_n_quantiles
    trial_row["sampler_adapter"] = sampler_adapter
    trial_row["tuner_searcher_tuning_framework"] = tuner_searcher_tuning_framework

    return trial_row


def generate_warm_starts_with_strategy(
    search_space: dict[str, FloatRange | IntRange | CategoricalRange],
    n_configs: int,
    random_state: int,
    objective_function: object,
    strategy: WarmStartStrategy,
) -> list[tuple[dict, float]]:
    """Dispatch warm start generation to the appropriate strategy.

    For GP strategies, ``n_configs`` random draws are used to seed the GP and
    ``n_configs`` GP-guided trials are returned, giving twice the evaluations
    under the hood.

    Args:
        search_space: Hyperparameter search space definition.
        n_configs: Configurations to generate (and to seed the GP when applicable).
        random_state: Random seed for reproducibility.
        objective_function: Surrogate model with a ``predict_batch`` method.
        strategy: One of ``"random"``, ``"gp_thompson_sampling"``,
            or ``"gp_expected_improvement"``.

    Returns:
        List of (configuration, performance) tuples.

    Raises:
        ValueError: If ``strategy`` is not a recognised ``WarmStartStrategy``.
    """
    if strategy == "random":
        return _generate_random_warm_starts(
            search_space=search_space,
            n_configs=n_configs,
            random_state=random_state,
            objective_function=objective_function,
        )
    elif strategy == "gp_thompson_sampling":
        return _generate_gp_warm_starts(
            search_space=search_space,
            n_initial_random=n_configs,
            n_gp_searches=n_configs,
            random_state=random_state,
            objective_function=objective_function,
            acquisition_strategy="TS",
        )
    elif strategy == "gp_expected_improvement":
        return _generate_gp_warm_starts(
            search_space=search_space,
            n_initial_random=n_configs,
            n_gp_searches=n_configs,
            random_state=random_state,
            objective_function=objective_function,
            acquisition_strategy="EI",
        )
    else:
        raise ValueError(f"Unknown warm start strategy: {strategy}")


def run_main_benchmark(
    experiment_configs: list[ExperimentConfig],
    n_repetitions: int,
    cache_path: str,
    run_start_str: str,
    base_random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Execute the core HPO benchmark loop across all experiment configurations.

    For each dataset the loop: initialises the surrogate objective, generates warm
    start configs once per (strategy, repetition) pair so all tuners share identical
    starting conditions, runs exactly ``n_ws + 1`` total trials (warm-starts plus one
    optimisation step), and retains only the final trial row. Results are incrementally
    persisted to disk after each trial to guard against mid-run failures.

    Args:
        experiment_configs: Pre-configured experiment instances, each binding a dataset
            to its search space, objective function, and set of tuners.
        n_repetitions: Independent repetitions per tuner-dataset combination.
        cache_path: Root directory for saving results and intermediate data.
        run_start_str: Unique run identifier used to namespace output paths.
        base_random_state: Base seed; each repetition uses ``base_random_state + rep``.

    Returns:
        DataFrame of all trial results with performance metrics and full metadata.
    """
    from hpobench.orchestration.meta_features import calculate_surrogate_metafeatures

    experiment_params = ExperimentParameters()
    n_warm_starts = experiment_params.n_warm_starts
    warm_start_strategies = experiment_params.warm_start_strategies

    logger.info("Running HPO benchmark...")

    incremental_data_path = os.path.join(cache_path, f"data/{run_start_str}")
    os.makedirs(incremental_data_path, exist_ok=True)

    raw_benchmark_data = pd.DataFrame()
    logger.info(f"Starting benchmark run with {len(experiment_configs)} experiment configurations")

    for config_idx, experiment_config in enumerate(experiment_configs, 1):
        dataset_name = experiment_config.dataset_identifier
        logger.info(f"[Config {config_idx}/{len(experiment_configs)}] Dataset: {dataset_name}")

        experiment_config.objective_function.initialize()

        for ws_idx, n_ws in enumerate(n_warm_starts, 1):
            logger.info(
                f"Warm start loop [{ws_idx}/{len(n_warm_starts)}] - "
                f"Generating {n_ws} warm start configurations for dataset: {dataset_name}"
            )

            for strategy in warm_start_strategies:
                logger.info(f"Using warm-start strategy: {strategy}")

                warm_start_configs_per_repetition = []
                for repetition in range(n_repetitions):
                    warm_start_configs = generate_warm_starts_with_strategy(
                        search_space=experiment_config.search_space,
                        n_configs=n_ws,
                        random_state=base_random_state + repetition,
                        objective_function=experiment_config.objective_function,
                        strategy=strategy,
                    )
                    warm_start_configs_per_repetition.append(warm_start_configs)

                logger.info(
                    f"Generated {len(warm_start_configs_per_repetition[0])} warm start configurations "
                    f"using {strategy} strategy."
                )

                for tuner in experiment_config.tuner_configurations:
                    logger.info(f"Loop Level | Tuner: {tuner}")
                    for repetition in range(n_repetitions):
                        logger.info(f"Loop Level | Repetition: {repetition}")
                        tune_start = datetime.now()

                        configs = [config for config, _ in warm_start_configs_per_repetition[repetition]]
                        performances = [perf for _, perf in warm_start_configs_per_repetition[repetition]]

                        schema = SurrogateMetafeaturesSchema()
                        surrogate_metafeatures = calculate_surrogate_metafeatures(
                            configs=configs,
                            performances=performances,
                            schema=schema,
                            search_space=experiment_config.search_space,
                            optimization_direction="minimize",
                        )
                        logger.info(
                            f"Calculated surrogate metafeatures for repetition {repetition} "
                            f"from {len(configs)} warm-start configs ({strategy}): {surrogate_metafeatures}"
                        )

                        historical_performance = tune(
                            performance_generator=experiment_config.objective_function,
                            tuner_config=tuner,
                            n_trials=n_ws + 1,
                            timeout=None,
                            params=experiment_config.search_space,
                            warm_start_configs=warm_start_configs_per_repetition[repetition],
                            random_state=base_random_state + repetition,
                        )

                        if len(historical_performance) != n_ws + 1:
                            raise ValueError(
                                f"Expected {n_ws + 1} total trials but got {len(historical_performance)}"
                            )

                        historical_performance = add_runtime(
                            experiment_log=historical_performance,
                            tune_start=tune_start,
                            performance_generator=experiment_config.objective_function,
                        )

                        # Retain only the final optimisation trial; warm-start rows are auxiliary.
                        historical_performance = historical_performance.tail(1).copy()
                        historical_performance = _annotate_trial_result(
                            trial_row=historical_performance,
                            experiment_config=experiment_config,
                            tuner=tuner,
                            repetition=repetition,
                            n_ws=n_ws,
                            strategy=strategy,
                            surrogate_metafeatures=surrogate_metafeatures,
                            aliases=aliases,
                        )

                        raw_benchmark_data = pd.concat(
                            [raw_benchmark_data, historical_performance], axis=0
                        )

                        data_path = os.path.join(cache_path, f"data/{run_start_str}")
                        os.makedirs(data_path, exist_ok=True)
                        raw_benchmark_data.to_csv(
                            os.path.join(data_path, "incremental_raw_benchmark_data.csv"),
                            index=False,
                        )

        experiment_config.objective_function = None
        gc.collect()

    final_data_path = os.path.join(cache_path, f"data/{run_start_str}")
    os.makedirs(final_data_path, exist_ok=True)
    final_filename = os.path.join(final_data_path, "raw_benchmark_data.csv")
    raw_benchmark_data.to_csv(final_filename, index=False)
    logger.info(f"Final raw benchmark data saved to {final_filename} ({len(raw_benchmark_data)} rows).")
    return raw_benchmark_data


def run_learning_to_rank_analysis(
    raw_benchmark_data: pd.DataFrame,
    schema: BenchmarkDataSchema,
    metafeatures_schema: SurrogateMetafeaturesSchema,
    synthetic_benchmark_id: str,
    ltr_config: LTRConfig,
    downsampling_sample_sizes: list[int],
    tuner_encoding_method: TunerEncoding = "ordinal",
    compute_pdp: bool = True,
    pdp_n_grid_points: int = 20,
    pdp_show_std: bool = True,
    output_dir: Path = Path("cache/ltr_results"),
) -> dict[str, LTRAnalysis]:
    """Fit, evaluate, and persist LTR models across predefined analysis partitions.

    Runs four analysis configurations covering all data, synthetic-train/real-test,
    synthetic-only, and real-only partitions. Each analysis fits an LTR model,
    computes evaluation metrics, optionally computes partial dependence plots,
    runs a downsampling study, and saves all artefacts to disk.

    Args:
        raw_benchmark_data: Benchmark trial results produced by ``run_main_benchmark``.
        schema: Column schema describing the benchmark data layout.
        metafeatures_schema: Schema for surrogate metafeatures.
        synthetic_benchmark_id: Identifier for synthetic benchmark data.
        ltr_config: Configuration for LTR training and evaluation.
        downsampling_sample_sizes: Training group counts at which to evaluate robustness.
        tuner_encoding_method: Algorithm identity encoding — ``"ordinal"`` or ``"one_hot"``.
        compute_pdp: Whether to compute rank-based partial dependence plots.
        pdp_n_grid_points: Grid resolution per feature for PDP computation.
        pdp_show_std: Whether to include standard deviation bands in PDP plots.
        output_dir: Root directory for saving analysis artefacts.

    Returns:
        Dictionary mapping analysis identifier to its fitted ``LTRAnalysis`` object.
    """
    analysis_configs = [
        AnalysisConfig(partition="all",       strategy="random"),
        AnalysisConfig(partition="all",       strategy="synthetic_train_real_test"),
        AnalysisConfig(partition="synthetic", strategy="random"),
        AnalysisConfig(partition="real",      strategy="random"),
    ]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    analyses: dict[str, LTRAnalysis] = {}

    for ac in analysis_configs:
        analysis = LTRAnalysis(
            schema=schema,
            metafeatures_schema=metafeatures_schema,
            synthetic_benchmark_id=synthetic_benchmark_id,
            ltr_config=ltr_config,
            partition=ac.partition,
            strategy=ac.strategy,
            tuner_encoding_method=tuner_encoding_method,
        )

        analysis.fit(raw_benchmark_data, k_values=ltr_config.k_values)
        analysis_id = analysis.analysis_identifier
        analyses[analysis_id] = analysis
        analysis.evaluate(k_values=ltr_config.k_values)

        if compute_pdp:
            analysis.compute_pdp(
                output_dir=output_dir / analysis_id,
                n_grid_points=pdp_n_grid_points,
                show_std=pdp_show_std,
            )

        analysis.compute_downsampling(
            sample_sizes=downsampling_sample_sizes,
            output_dir=output_dir / analysis_id,
        )

        analysis.save(output_dir / analysis_id)

    summary_df = pd.DataFrame([a.summary() for a in analyses.values()])
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    return analyses


def run_and_analyze_main_benchmark(
    benchmarks: list[
        Literal[
            "lcbench",
            "rbv2_aknn",
            "LCBench-L",
            "LCBench-H",
            "LCBench-A",
            "rbv2_aknn-L",
            "rbv2_aknn-H",
            "rbv2_aknn-A",
            "synthetic_tabular",
        ]
    ],
    tuning_configurations: list[TunerConfig],
    base_random_state: int,
    schema: BenchmarkDataSchema,
    cache_path: str,
    run_start_str: str,
    experiment_params: ExperimentParameters,
    results_dir: Path,
    downsampling_percentages: list[float],
    max_n_instances_per_benchmark: int = 10,
    n_repetitions: int = 10,
    datasets_per_benchmark: Optional[list[list[str]]] = None,
) -> None:
    """Run the full benchmark pipeline: data generation, HPO trials, and LTR analysis.

    Each dataset receives exactly one optimisation trial after its warm-start
    configurations. Downsampling study evaluates robustness across percentages of
    the full ranking group set.

    Args:
        benchmarks: Benchmark names to include in the run.
        tuning_configurations: HPO algorithm configurations to compare.
        base_random_state: Seed for reproducible experiments.
        schema: Column schema for result organisation and LTR training.
        cache_path: Root directory for all output data and analysis artefacts.
        run_start_str: Unique run identifier used to namespace output paths.
        experiment_params: Experiment parameters including LTR settings.
        results_dir: Output directory for LTR analysis results.
        downsampling_percentages: Percentages of ranking groups for downsampling study (e.g., [0.1, 0.5, 1.0]).
        max_n_instances_per_benchmark: Maximum dataset instances per benchmark.
        n_repetitions: Independent repetitions per tuner-dataset combination.
        datasets_per_benchmark: Optional per-benchmark dataset identifier lists.
    """
    experiment_configs = load_experiment_configs(
        benchmarks=benchmarks,
        tuning_configurations=tuning_configurations,
        max_n_instances_per_benchmark=max_n_instances_per_benchmark,
        datasets_per_benchmark=datasets_per_benchmark,
    )

    raw_benchmark_data = run_main_benchmark(
        experiment_configs=experiment_configs,
        n_repetitions=n_repetitions,
        base_random_state=base_random_state,
        cache_path=cache_path,
        run_start_str=run_start_str,
    )

    n_groups = raw_benchmark_data.groupby(schema.rank_group_cols).ngroups
    downsampling_sample_sizes = sorted(set(
        int(n_groups * pct) for pct in downsampling_percentages if 0 < pct <= 1.0
    ) | {n_groups})

    logger.info("Running learning-to-rank analysis on benchmark results")

    ltr_config = LTRConfig()
    metafeatures_schema = SurrogateMetafeaturesSchema()
    synthetic_params = SyntheticGenerationParameters()
    
    run_learning_to_rank_analysis(
        raw_benchmark_data=raw_benchmark_data,
        schema=schema,
        metafeatures_schema=metafeatures_schema,
        synthetic_benchmark_id=synthetic_params.benchmark_identifier,
        ltr_config=ltr_config,
        downsampling_sample_sizes=downsampling_sample_sizes,
        tuner_encoding_method=experiment_params.tuner_encoding_method,
        compute_pdp=experiment_params.compute_pdp,
        pdp_n_grid_points=experiment_params.pdp_n_grid_points,
        pdp_show_std=experiment_params.pdp_show_std,
        output_dir=results_dir,
    )
