# HPO Benchmark

A comprehensive benchmarking framework for comparing various hyperparameter optimization (HPO) algorithms and analyzing their performance characteristics across different benchmark suites.

## Overview

The HPO Benchmark package provides a systematic evaluation framework for hyperparameter optimization algorithms, focusing on:
- Comparing different HPO algorithms (Optuna, Scikit-Optimize, Conformal Prediction methods)
- Analyzing performance across multiple benchmark suites
- Evaluating conformal prediction approaches for uncertainty quantification
- Providing statistical analysis and visualization of results

## Main Components

### Core Modules

- **`hpobench.tune`**: Main tuning interface supporting multiple HPO frameworks
- **`hpobench.prepare`**: Benchmark configuration and setup utilities
- **`hpobench.process`**: Data processing and result handling
- **`hpobench.report`**: Analysis, visualization, and reporting tools
- **`hpobench.generation`**: Objective function generators and synthetic benchmarks

### Configuration

- **`hpobench.config.config`**: Main configuration parameters and tuning setups
- **`hpobench.config.types`**: Type definitions for experiments and configurations
- **`hpobench.config.benchmark_data`**: Benchmark-specific parameter spaces

### Integrations

- **`hpobench.syne_tune_integration`**: Syne-Tune framework integration for advanced methods

## File Structure

```
hpo-benchmark/
├── run.py                          # Main execution script
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Package configuration
├── hpobench/                       # Main package
│   ├── __init__.py
│   ├── tune.py                     # HPO algorithm implementations
│   ├── prepare.py                  # Benchmark setup
│   ├── process.py                  # Data processing
│   ├── plot.py                     # Visualization utilities
│   ├── utils.py                    # General utilities
│   ├── syne_tune_integration.py    # Syne-Tune integration
│   ├── config/                     # Configuration files
│   │   ├── config.py              # Main configuration
│   │   ├── types.py               # Type definitions
│   │   ├── benchmark_data.py      # Benchmark specifications
│   │   └── utils.py               # Config utilities
│   ├── generation/                 # Objective function generation
│   │   ├── generate.py            # Objective generators
│   │   └── black_box_functions.py # Synthetic functions
│   └── report/                     # Analysis and reporting
│       ├── orchestrate.py         # Main orchestration
│       ├── analyze.py             # Statistical analysis
│       ├── metrics.py             # Performance metrics
│       └── utils.py               # Report utilities
├── tests/                          # Test suite
├── cache/                          # Experiment cache and results
├── yahpo_bench_data/              # YAHPO benchmark data
└── jahs_bench_data/               # JAHS-Bench-201 data
```

## Running Experiments

### Basic Usage

Execute the main benchmark script:

```bash
python run.py
```

### Configuration

The `run.py` script contains a `run_sections` dictionary that controls which analyses to execute:

```python
run_sections = {
    "run_coverage_analysis": False,
    "run_sampler_variation_analysis": False,
    "run_architecture_variation_analysis": False,
    "run_external_tuning_analysis": True,
    "run_preconformal_comparison_analysis": False,
    "run_static_analysis": False,
}
```

Set the desired sections to `True` to run specific analyses.

### Key Parameters

The main configuration parameters are defined in `hpobench/config/config.py`:

- **`N_TRIALS`** (default: 100): Number of optimization trials per experiment
- **`N_WARM_STARTS`** (default: 15): Number of random initial evaluations
- **`N_REPETITIONS_PER_TUNER_CONFIG`** (default: 1): Number of repetitions per configuration
- **`TIMEOUT`** (default: None): Time limit per experiment in seconds
- **`DEFAULT_MAX_N_INSTANCES`** (default: 20): Maximum benchmark instances per suite
- **`BASE_RANDOM_STATE`** (default: 42): Random seed for reproducibility

### Analysis Types

1. **Coverage Analysis**: Evaluates conformal prediction coverage properties
2. **Sampler Variation Analysis**: Compares different sampling strategies
3. **Architecture Variation Analysis**: Tests different model architectures
4. **External Tuning Analysis**: Benchmarks against external HPO methods
5. **Preconformal Comparison Analysis**: Analyzes pre-conformal vs conformal methods
6. **Static Analysis**: Evaluates estimator performance without optimization

### HPO Algorithm Support

The framework supports multiple HPO algorithms:

- **Optuna**: TPE, Random, CMA-ES, GP samplers
- **Scikit-Optimize**: GP, Random Forest, Gradient Boosting
- **Conformal Prediction**: Quantile-based conformal methods
- **Syne-Tune**: Conformal Quantile Regression methods

## Benchmark Suites

### YAHPO Gym

**Description**: Yet Another Hyperparameter Optimization Gym provides surrogate models for various machine learning benchmarks.

**Supported Benchmarks**:
- `rbv2_aknn`: Approximate Nearest Neighbours on OpenML datasets
- `lcbench`: Learning Curve Benchmark

**Setup Instructions**:
1. Install the YAHPO Gym package (included in requirements.txt)
2. **Manual Data Setup Required**: Download the YAHPO benchmark data from the forked repository at: https://github.com/rick12000/yahpo_data_snapshot
3. Extract the data folders into the `yahpo_bench_data/` folder at the root of this repository
4. The folder structure should contain subdirectories like:
   - `yahpo_bench_data/rbv2_aknn/`
   - `yahpo_bench_data/lcbench/`
   - `yahpo_bench_data/iaml_*/`

### JAHS-Bench-201

**Description**: Joint Architecture and Hyperparameter Search Benchmark for neural architecture search.

**Supported Datasets**:
- CIFAR-10
- Fashion-MNIST
- Colorectal Histology

**Setup Instructions**:
1. Install the JAHS-Bench package (included in requirements.txt)
2. **Automatic Data Download**: The benchmark data will be automatically downloaded to the `jahs_bench_data/` folder upon first run, or execute `python -m jahs_bench.download --target surrogates` to load all benchmark datasets at once

### Synthetic Benchmarks

**Description**: Mathematical optimization functions for controlled experiments.

**Available Functions**:
- Rastrigin
- Shekel
- Weierstrass
- Griewank
- Ackley

**Setup**: No additional setup required - functions are implemented directly in the package.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rick12000/hpo-benchmark.git
   cd hpo-benchmark
   ```

2. **Install SWIG (Required for SMAC)**:
   SWIG is required to build the `pyrfr` dependency for SMAC. Install it using conda:
   ```bash
   conda install swig
   ```

   Note: SWIG must be installed in the same environment where you're installing the package dependencies.

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install the package:
   ```bash
   pip install -e .
   ```

5. Set up benchmark data (see Benchmark Suites section above)

## Results and Analysis

- **Cache Directory**: All results are stored in the `cache/` folder with timestamped subdirectories
- **Plots**: Generated visualizations are saved in `cache/plots/`
- **Data**: Raw experiment data is stored in `cache/data/`
- **Logs**: Execution logs are maintained in `cache/logs/`

## Requirements

- Python >= 3.10, < 3.11
- Key dependencies: pandas, matplotlib, scikit-learn, optuna, scikit-optimize, yahpo_gym, jahs-bench, syne_tune

See `requirements.txt` for complete dependency list.
