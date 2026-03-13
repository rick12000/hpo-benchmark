import pytest
import pandas as pd
from pathlib import Path
from hpobench.learning_to_rank.scaling import compute_downsampling_curve, plot_downsampling_curve
from hpobench.config.types import DownsamplingResults

def test_downsampling_computation(fitted_ltr_analysis, tmp_path):
    """Test downsampling curve computation and plotting."""
    train_val_data = pd.concat([fitted_ltr_analysis.train_data, fitted_ltr_analysis.val_data], ignore_index=True)
    test_data = fitted_ltr_analysis.test_data
    feature_cols = fitted_ltr_analysis.feature_cols
    config = fitted_ltr_analysis.config
    schema = fitted_ltr_analysis.schema
    
    # We have ~12 groups in train_val_data (from preprocessing_raw_data)
    sample_sizes_val = [2, 5, 100]
    
    results = compute_downsampling_curve(
        train_val_data=train_val_data,
        test_data=test_data,
        feature_cols=feature_cols,
        requested_sample_sizes=sample_sizes_val,
        ltr_config=config,
        schema=schema,
        analysis_identifier="test_downsampling"
    )
    
    assert isinstance(results, DownsamplingResults)
    # Check that sample sizes are correct (sorted and capped/appended)
    # The function sorts unique valid sizes and adds total available if not present.
    # Total available is 12 (0.8 * 16).
    # Valid requested: 2, 5.
    # Result: 2, 5, 12.
    assert len(results.sample_sizes) >= 2
    assert results.sample_sizes[-1] <= 16
    
    assert 'precision@1' in results.metrics
    assert len(results.metrics['precision@1']) == len(results.sample_sizes)
    assert len(results.n_train_groups) == len(results.sample_sizes)
    assert len(results.n_val_groups) == len(results.sample_sizes)
    
    # Plotting
    plot_downsampling_curve(downsampling_results=results, output_path=tmp_path / 'curve.png')
    assert (tmp_path / 'curve.png').exists()
