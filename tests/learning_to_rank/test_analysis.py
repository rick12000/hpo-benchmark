import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json
from hpobench.learning_to_rank.analysis import LTRAnalysis
from hpobench.config.types import PartialDependenceResults, DownsamplingResults

def test_ltr_analysis_fit_evaluate(ltr_analysis, preprocessing_raw_data, tmp_path):
    """Test fit and evaluate workflow of LTRAnalysis."""
    k_values_val = (1, 3)
    # Fit the analysis
    ltr_analysis.fit(raw_data=preprocessing_raw_data, k_values=k_values_val)
    
    # Check that data was split and processed
    assert ltr_analysis.train_data is not None
    assert ltr_analysis.val_data is not None
    assert ltr_analysis.test_data is not None
    assert len(ltr_analysis.feature_cols) > 0
    assert ltr_analysis.ltr_model is not None
    assert ltr_analysis.baseline_ranker is not None
    
    # Check evaluate
    metrics = ltr_analysis.evaluate(k_values=k_values_val, output_dir=tmp_path)
    
    assert 'ltr_metrics' in metrics
    assert 'naive_metrics' in metrics
    assert 'n_test' in metrics
    assert metrics['n_test'] > 0
    
    # Check saved metrics
    metrics_file = tmp_path / 'metrics.json'
    assert metrics_file.exists()
    with open(metrics_file) as f:
        saved_metrics = json.load(f)
    assert saved_metrics['n_test'] == metrics['n_test']
    assert 'precision@1' in saved_metrics['ltr_metrics']

def test_ltr_analysis_pdp_shap_downsampling(ltr_analysis, preprocessing_raw_data, tmp_path):
    """Test PDP, SHAP, and downsampling computation."""
    k_values_val = (1, 3)
    ltr_analysis.fit(raw_data=preprocessing_raw_data, k_values=k_values_val)
    
    # Test PDP
    n_grid_points_val = 5
    n_bootstrap_val = 10
    pdp_results = ltr_analysis.compute_pdp(output_dir=tmp_path / 'pdp', n_grid_points=n_grid_points_val, n_bootstrap=n_bootstrap_val)
    assert isinstance(pdp_results, PartialDependenceResults)
    assert len(pdp_results.results) > 0
    assert (tmp_path / 'pdp').exists()
    assert list((tmp_path / 'pdp').glob('*.png'))
    
    # Test SHAP
    top_k_val = 5
    sample_size_val = 5
    shap_out = ltr_analysis.compute_shap(output_dir=tmp_path / 'shap', top_k=top_k_val, sample_size=sample_size_val)
    assert 'shap_results' in shap_out
    assert 'summary' in shap_out
    assert (tmp_path / 'shap').exists()
    assert (tmp_path / 'shap' / 'feature_importance.csv').exists()
    assert (tmp_path / 'shap' / 'importance_bar.png').exists()
    
    # Test Downsampling
    # We need enough groups for downsampling. preprocessing_raw_data has 2 datasets * 2 strategies * 2 n_warm * 2 reps = 16 groups.
    # Split is 0.6 train, 0.2 val, 0.2 test. So roughly 12 groups for train+val.
    sample_sizes_val = [2, 5, 100] # 100 is larger than available, should be handled
    down_results = ltr_analysis.compute_downsampling(sample_sizes=sample_sizes_val, output_dir=tmp_path / 'downsampling')
    
    assert isinstance(down_results, DownsamplingResults)
    assert len(down_results.sample_sizes) > 0
    # Check that 100 was capped or handled (it usually appends the full size)
    assert down_results.sample_sizes[-1] <= 16 
    assert (tmp_path / 'downsampling').exists()
    assert (tmp_path / 'downsampling' / 'downsampling_curve.csv').exists()
    assert (tmp_path / 'downsampling' / 'downsampling_curve.png').exists()

def test_ltr_analysis_strategy_validation(ltr_analysis, preprocessing_raw_data):
    """Test validation of split strategies."""
    # Test synthetic_train_real_test requirements
    ltr_analysis.strategy = 'synthetic_train_real_test'
    
    # Should work with partition='all' and data having both synthetic and real
    # preprocessing_raw_data has 'synthetic_tabular' and 'lcbench' (real)
    ltr_analysis._validate_strategy_eligibility(raw_data=preprocessing_raw_data)
    
    # Fail if partition is not 'all'
    ltr_analysis.partition = 'synthetic'
    with pytest.raises(ValueError, match="requires partition=all"):
        ltr_analysis._validate_strategy_eligibility(raw_data=preprocessing_raw_data)
    
    # Fail if no synthetic data
    ltr_analysis.partition = 'all'
    real_only = preprocessing_raw_data[preprocessing_raw_data['benchmark_identifier'] != 'synthetic_tabular']
    with pytest.raises(ValueError, match="requires synthetic data"):
        ltr_analysis._validate_strategy_eligibility(raw_data=real_only)
        
    # Fail if no real data
    synthetic_only = preprocessing_raw_data[preprocessing_raw_data['benchmark_identifier'] == 'synthetic_tabular']
    with pytest.raises(ValueError, match="requires real data"):
        ltr_analysis._validate_strategy_eligibility(raw_data=synthetic_only)

def test_ltr_analysis_not_fitted_errors(ltr_analysis, tmp_path):
    """Ensure methods raise RuntimeError if called before fit."""
    with pytest.raises(RuntimeError, match="call fit"):
        ltr_analysis.evaluate(k_values=(1,))
        
    with pytest.raises(RuntimeError, match="call fit"):
        ltr_analysis.compute_pdp()
        
    with pytest.raises(RuntimeError, match="call fit"):
        ltr_analysis.compute_shap()
        
    with pytest.raises(RuntimeError, match="call fit"):
        ltr_analysis.compute_downsampling(sample_sizes=[5])
