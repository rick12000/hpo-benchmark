import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from hpobench.learning_to_rank.explainability import (
    compute_shap_values,
    shap_importance_summary,
    compute_partial_dependence,
    plot_shap_importance,
    plot_shap_beeswarm,
    plot_partial_dependence,
    run_shap_analysis,
)
from hpobench.config.types import SharpResults, PartialDependenceResults

def test_shap_computation(fitted_ltr_analysis, tmp_path):
    """Test SHAP value computation and summary."""
    model = fitted_ltr_analysis.ltr_model.booster
    data = fitted_ltr_analysis.test_data
    feature_cols = fitted_ltr_analysis.feature_cols
    group_col = fitted_ltr_analysis.schema.ranking_group_id_col

    shap_results = compute_shap_values(
        model=model, data=data, feature_cols=feature_cols,
        group_col=group_col, sample_size=10,
    )
    
    assert isinstance(shap_results, SharpResults)
    assert shap_results.shap_values.shape == (len(data), len(feature_cols))
    assert shap_results.feature_names == feature_cols
    
    summary = shap_importance_summary(shap_results=shap_results)
    assert isinstance(summary, pd.DataFrame)
    assert 'feature' in summary.columns
    assert 'mean_abs_shap' in summary.columns
    assert len(summary) == len(feature_cols)
    assert summary['mean_abs_shap'].is_monotonic_decreasing
    
    plot_shap_importance(shap_results=shap_results, output_path=tmp_path / 'importance.png')
    assert (tmp_path / 'importance.png').exists()
    
    plot_shap_beeswarm(shap_results=shap_results, output_path=tmp_path / 'beeswarm.png')
    assert (tmp_path / 'beeswarm.png').exists()


def test_run_shap_analysis(fitted_ltr_analysis, tmp_path):
    """Test the high-level run_shap_analysis function."""
    model = fitted_ltr_analysis.ltr_model.booster
    data = fitted_ltr_analysis.test_data
    feature_cols = fitted_ltr_analysis.feature_cols
    group_col = fitted_ltr_analysis.schema.ranking_group_id_col

    results = run_shap_analysis(
        model=model, data=data, feature_cols=feature_cols,
        group_col=group_col,
        output_dir=tmp_path,
        sample_size=10,
    )
    
    assert 'shap_results' in results
    assert 'summary' in results
    assert (tmp_path / 'feature_importance.csv').exists()
    assert (tmp_path / 'importance_bar.png').exists()
    assert (tmp_path / 'importance_beeswarm.png').exists()

def test_pdp_computation(fitted_ltr_analysis, tmp_path):
    """Test PDP computation and plotting."""
    model = fitted_ltr_analysis.ltr_model.booster
    data = fitted_ltr_analysis.test_data
    feature_cols = fitted_ltr_analysis.feature_cols
    schema = fitted_ltr_analysis.schema
    n_grid_points_val = 5
    n_bootstrap_val = 10
    
    pdp_results = compute_partial_dependence(
        model=model,
        data=data,
        feature_cols=feature_cols,
        group_col=schema.ranking_group_id_col,
        tuner_col=schema.tuner_col,
        n_grid_points=n_grid_points_val,
        n_bootstrap=n_bootstrap_val
    )
    
    assert isinstance(pdp_results, PartialDependenceResults)
    assert len(pdp_results.results) > 0
    
    # Check structure of all results
    for res in pdp_results.results:
        assert res.feature_name in feature_cols
        assert len(res.x_values) <= n_grid_points_val
        assert len(res.rank_means) == len(res.x_values)
    
    # Plotting
    plot_partial_dependence(pdp_results=pdp_results, output_dir=tmp_path)
    assert list(tmp_path.glob('pdp_*.png'))
