"""Tests for preprocessing utilities."""

import numpy as np

from pySPFM.cli._preprocessing import remove_regressors


class TestRemoveRegressors:
    """Tests for the remove_regressors function."""

    def test_single_echo_single_regressor(self):
        """Test regressor removal with single echo and single regressor."""
        np.random.seed(42)
        n_scans = 100
        n_voxels = 10

        # Create data with a known confound
        confound = np.sin(np.linspace(0, 4 * np.pi, n_scans))
        data = np.random.randn(n_scans, n_voxels)
        # Add confound to data
        data += confound[:, np.newaxis] * 2

        regressors = confound.reshape(-1, 1)
        cleaned = remove_regressors(data, regressors, n_scans)

        assert cleaned.shape == data.shape
        # The variance should be reduced after removing the confound
        assert np.var(cleaned) < np.var(data)

    def test_single_echo_multiple_regressors(self):
        """Test regressor removal with multiple regressors."""
        np.random.seed(42)
        n_scans = 100
        n_voxels = 10
        n_regressors = 3

        data = np.random.randn(n_scans, n_voxels)
        regressors = np.random.randn(n_scans, n_regressors)

        # Add regressors to data
        for i in range(n_regressors):
            data += regressors[:, i : i + 1] * (i + 1)

        cleaned = remove_regressors(data, regressors, n_scans)

        assert cleaned.shape == data.shape

    def test_multi_echo(self):
        """Test regressor removal with multi-echo data."""
        np.random.seed(42)
        n_scans = 50
        n_echoes = 3
        n_voxels = 10

        # Multi-echo data has echoes concatenated
        data = np.random.randn(n_scans * n_echoes, n_voxels)
        regressors = np.random.randn(n_scans, 2)

        cleaned = remove_regressors(data, regressors, n_scans, n_echoes)

        assert cleaned.shape == data.shape
        assert cleaned.shape[0] == n_scans * n_echoes

    def test_1d_regressor_handling(self):
        """Test that 1D regressors are handled correctly."""
        np.random.seed(42)
        n_scans = 50
        n_voxels = 5

        data = np.random.randn(n_scans, n_voxels)
        # 1D regressor (not 2D)
        regressors = np.random.randn(n_scans)

        cleaned = remove_regressors(data, regressors, n_scans)

        assert cleaned.shape == data.shape

    def test_preserves_mean(self):
        """Test that the mean signal level is approximately preserved."""
        np.random.seed(42)
        n_scans = 100
        n_voxels = 10

        # Create data with known mean
        data = np.random.randn(n_scans, n_voxels) + 100
        regressors = np.random.randn(n_scans, 2)

        cleaned = remove_regressors(data, regressors, n_scans)

        # Mean should be approximately preserved (intercept not removed)
        np.testing.assert_allclose(np.mean(cleaned, axis=0), np.mean(data, axis=0), rtol=0.1)

    def test_does_not_modify_original(self):
        """Test that the original data array is not modified."""
        np.random.seed(42)
        n_scans = 50
        n_voxels = 5

        data = np.random.randn(n_scans, n_voxels)
        data_copy = data.copy()
        regressors = np.random.randn(n_scans, 2)

        _ = remove_regressors(data, regressors, n_scans)

        np.testing.assert_array_equal(data, data_copy)

    def test_removes_correlated_signal(self):
        """Test that correlated signals are effectively removed."""
        np.random.seed(42)
        n_scans = 100
        n_voxels = 5

        # Create a known confound signal
        regressor = np.sin(np.linspace(0, 4 * np.pi, n_scans)).reshape(-1, 1)

        # Create data that is highly correlated with the regressor
        noise = np.random.randn(n_scans, n_voxels) * 0.1
        data = regressor * np.random.randn(1, n_voxels) * 5 + noise

        # Compute correlation before cleaning
        corr_before = []
        for i in range(n_voxels):
            corr = np.corrcoef(data[:, i], regressor.ravel())[0, 1]
            corr_before.append(np.abs(corr))
        corr_before = np.array(corr_before)

        cleaned = remove_regressors(data, regressor, n_scans)

        # Compute correlation after cleaning
        corr_after = []
        for i in range(n_voxels):
            corr = np.corrcoef(cleaned[:, i], regressor.ravel())[0, 1]
            corr_after.append(np.abs(corr))
        corr_after = np.array(corr_after)

        # Correlation should be much lower after cleaning
        assert np.all(corr_after < corr_before)
        assert np.all(corr_after < 0.1)  # Should be nearly uncorrelated
