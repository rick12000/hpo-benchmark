# HPO Benchmarker

A general benchmarking framework for evaluating hyperparameter optimization (HPO) algorithms.

This repository serves as a reproducible track record for all analysis and figures used in the (TODO) paper. It is not intended as a standalone benchmarking utility and is not coded with the rigor of one.

For source code pertaining to conformalized hyperparameter optimization, refer to the [confopt package](https://github.com/rick12000/confopt).

## Installation

### 1. Repository:
1. Clone repository:
   ```bash
   git clone https://github.com/rick12000/hpo-benchmark.git
   cd hpo-benchmark
   ```

2. Install as package:
   ```bash
   pip install .
   ```

### 2. SMAC

`smac` is not a package dependancy of the `hpobench` package, due to incompatibility issues, but it is required to run SMAC benchmarks.

To resolve this, you can clone the below fork with minor edits to SMAC's `ConfigSpace` dependancy:
   ```bash
   git clone https://github.com/rick12000/SMAC3-ConfigSpace-Amend
   cd SMAC3-ConfigSpace-Amend
   ```

And install it in your environment directly by navigating to it while your python environment is active and running:
   ```bash
   pip install .
   ```

**NOTE**:
SWIG is required to build the `pyrfr` dependency for SMAC. Install it using conda:
   ```bash
   conda install swig
   ```
SWIG must be installed in the same environment where you're installing the package dependencies.

### 3. Confopt

To run confopt benchmarks, you must install the confopt package. For reproducible results that align to the paper, clone and install confopt from this branch (TODO):


Failing that, confopt can be installed from [pypi](https://pypi.org/project/confopt/) using:
   ```bash
   pip install confopt
   ```
The closest version to the static branch used for analysis is 2.0.0.

### 4. Benchmark Environment Setup

#### YAHPO Gym

(Currently limited to non-hierarchical benchmarks).

**Setup Instructions**:
1. Install the YAHPO Gym package (included in requirements.txt)
2. **Manual Data Setup Required**: Download the YAHPO benchmark data from the forked repository at: https://github.com/rick12000/yahpo_data_snapshot
3. Extract the data folders into the `yahpo_bench_data/` folder at the root of this repository
4. The folder structure should contain subdirectories like:
   - `yahpo_bench_data/rbv2_aknn/`
   - `yahpo_bench_data/lcbench/`
   - `yahpo_bench_data/iaml_*/`

#### JAHS-Bench-201

1. Install JAHS-Bench package (included in requirements.txt)
2. Data downloads automatically to `jahs_bench_data/` on first run
3. Or manually: `python -m jahs_bench.download --target surrogates`



## Running Experiments

### Entry Point

Execute benchmarks via the main script:

```bash
python run.py
```

### Experiment Configuration
- Algorithm agnostic parameters can be set in `constants.py`
- Algorithm configurations are set up in `tuner_configurations.py`


### Experiment Components

You can specify which types of analysis to run in `run.py` using the `run_sections` dictionary:

```python
run_sections = {
    "run_coverage_analysis": False,
    "run_sampler_variation_analysis": False,
    "run_architecture_variation_analysis": False,
    "run_external_tuning_analysis": True,
    "run_heteroscedastic_external_tuning_analysis": False,
    "run_skew_external_tuning_analysis": False,
    "run_preconformal_comparison_analysis": False,
    "run_static_analysis": False,
    "run_quantile_count_comparison": False,
    "run_search_tuning_effect_comparison": False,
}
```

## Supported HPO Packages:

- **Optuna**: TPE, CMA-ES, GP, Random
- **scikit-optimize**: GP, RF, GBRT
- **SMAC**: RF (limited/experimental support)
- **ConfOpt**: All surrogates
- **syne-tune**: CQR (limited/experimental support)
