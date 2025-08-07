"""Tests for documentation examples and content validation."""
import re
import pytest
import numpy as np
from pySPFM.deconvolution import stability_selection
from pySPFM.deconvolution.hrf_generator import HRFMatrix


class TestDocumentationExamples:
    """Test examples and concepts mentioned in documentation."""

    def test_auc_calculation_concept(self):
        """Test that AUC calculation works as described in documentation.
        
        The documentation explains that AUC serves as a proxy for how likely
        a time point is to have a non-zero coefficient across all possible lambdas.
        """
        # Create simple test data
        n_scans = 100
        n_echoes = 3
        n_timepoints = 200
        
        # Create a simple HRF matrix
        hrf_matrix = HRFMatrix(
            TR=2.0,
            TE=[14, 28, 42],
            n_scans=n_scans,
            block_model=False
        )
        
        # Create synthetic data with some signal
        np.random.seed(42)
        data = np.random.randn(n_scans * n_echoes, n_timepoints)
        
        # Add some signal at specific timepoints
        signal_timepoints = [50, 100, 150]
        for tp in signal_timepoints:
            data[:, tp] += np.random.randn(n_scans * n_echoes) * 2
        
        # Run stability selection
        n_lambdas = 10
        n_surrogates = 5  # Small number for test speed
        
        auc = stability_selection.stability_selection(
            hrf_matrix, data, n_lambdas, n_surrogates
        )
        
        # Verify AUC properties mentioned in documentation
        assert auc.shape == (n_timepoints,), "AUC should have one value per timepoint"
        assert np.all(auc >= 0), "AUC values should be non-negative"
        assert np.all(auc <= 1), "AUC values should be <= 1 (probability)"
        
        # As mentioned in docs: "By definition, the AUC time series cannot have zero values"
        # (this is only true if entire regularization path is explored)
        # For our test with limited lambdas, we just check they're reasonable
        assert np.mean(auc) > 0, "AUC values should generally be positive"

    def test_auc_thresholding_concept(self):
        """Test AUC thresholding as described in documentation.
        
        Documentation mentions using 95th percentile of deep white matter
        AUC values as threshold.
        """
        # Simulate AUC values for different brain regions
        np.random.seed(42)
        
        # Simulate "deep white matter" AUC values (should be lower)
        white_matter_auc = np.random.beta(2, 8, 1000)  # Skewed toward lower values
        
        # Simulate "active region" AUC values (should be higher)
        active_region_auc = np.random.beta(6, 4, 200)  # Skewed toward higher values
        
        # Calculate 95th percentile threshold as mentioned in docs
        threshold_95 = np.percentile(white_matter_auc, 95)
        
        # Also test 99th percentile mentioned in docs
        threshold_99 = np.percentile(white_matter_auc, 99)
        
        # Verify thresholds are reasonable
        assert 0 < threshold_95 < 1, "95th percentile threshold should be between 0 and 1"
        assert 0 < threshold_99 < 1, "99th percentile threshold should be between 0 and 1"
        assert threshold_99 > threshold_95, "99th percentile should be higher than 95th"
        
        # Apply thresholding
        thresholded_white_matter = white_matter_auc > threshold_95
        thresholded_active = active_region_auc > threshold_95
        
        # Verify that thresholding gives expected results
        # (more active regions should survive thresholding)
        white_matter_survival_rate = np.mean(thresholded_white_matter)
        active_region_survival_rate = np.mean(thresholded_active)
        
        assert white_matter_survival_rate <= 0.05, "~5% of white matter should survive 95th percentile threshold"
        assert active_region_survival_rate > white_matter_survival_rate, "Active regions should survive thresholding better"
        assert active_region_survival_rate < 1.0, "Not all active regions should survive thresholding"