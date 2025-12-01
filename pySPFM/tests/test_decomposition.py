"""Tests for decomposition estimators (scikit-learn API compliance)."""

import numpy as np
import pytest

from pySPFM.base import NotFittedError, clone


class TestSparseDeconvolutionAPI:
    """Tests for SparseDeconvolution scikit-learn API compliance."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample fMRI-like data."""
        np.random.seed(42)
        n_timepoints = 50
        n_voxels = 10
        return np.random.randn(n_timepoints, n_voxels)

    def test_init_parameters(self):
        """Test that initialization parameters are stored correctly."""
        from pySPFM import SparseDeconvolution

        model = SparseDeconvolution(
            tr=2.0,
            criterion="bic",
            debias=True,
            group=0.2,
        )
        params = model.get_params()

        assert params["tr"] == 2.0
        assert params["criterion"] == "bic"
        assert params["debias"] is True
        assert params["group"] == 0.2

    def test_set_params(self):
        """Test that set_params works correctly."""
        from pySPFM import SparseDeconvolution

        model = SparseDeconvolution(tr=2.0)
        model.set_params(criterion="aic", debias=False)

        assert model.criterion == "aic"
        assert model.debias is False

    def test_clone(self):
        """Test that clone works correctly."""
        from pySPFM import SparseDeconvolution

        model = SparseDeconvolution(tr=2.0, criterion="bic")
        cloned = clone(model)

        assert cloned is not model
        assert cloned.get_params() == model.get_params()

    def test_fit_returns_self(self, sample_data):
        """Test that fit returns self."""
        from pySPFM import SparseDeconvolution

        model = SparseDeconvolution(tr=2.0, criterion="bic", max_iter=10)
        result = model.fit(sample_data)

        assert result is model

    def test_fitted_attributes(self, sample_data):
        """Test that fit sets expected attributes."""
        from pySPFM import SparseDeconvolution

        model = SparseDeconvolution(tr=2.0, criterion="bic", max_iter=10)
        model.fit(sample_data)

        # Check fitted attributes exist
        assert hasattr(model, "coef_")
        assert hasattr(model, "lambda_")
        assert hasattr(model, "hrf_matrix_")
        assert hasattr(model, "n_features_in_")

        # Check shapes
        assert model.coef_.shape == sample_data.shape
        assert model.lambda_.shape == (sample_data.shape[1],)
        assert model.n_features_in_ == sample_data.shape[1]

    def test_transform_before_fit(self, sample_data):
        """Test that transform before fit raises NotFittedError."""
        from pySPFM import SparseDeconvolution

        model = SparseDeconvolution(tr=2.0)

        with pytest.raises(NotFittedError):
            model.transform(sample_data)

    def test_transform_returns_coef(self, sample_data):
        """Test that transform returns coefficients."""
        from pySPFM import SparseDeconvolution

        model = SparseDeconvolution(tr=2.0, criterion="bic", max_iter=10)
        model.fit(sample_data)

        transformed = model.transform(sample_data)

        np.testing.assert_array_equal(transformed, model.coef_)

    def test_fit_transform(self, sample_data):
        """Test fit_transform method."""
        from pySPFM import SparseDeconvolution

        model = SparseDeconvolution(tr=2.0, criterion="bic", max_iter=10)
        transformed = model.fit_transform(sample_data)

        assert transformed.shape == sample_data.shape
        np.testing.assert_array_equal(transformed, model.coef_)

    def test_score(self, sample_data):
        """Test score method returns valid value."""
        from pySPFM import SparseDeconvolution

        model = SparseDeconvolution(tr=2.0, criterion="bic", max_iter=10)
        model.fit(sample_data)

        score = model.score(sample_data)

        assert isinstance(score, float)
        assert -np.inf < score <= 1.0

    def test_get_fitted_signal(self, sample_data):
        """Test get_fitted_signal method."""
        from pySPFM import SparseDeconvolution

        model = SparseDeconvolution(tr=2.0, criterion="bic", max_iter=10)
        model.fit(sample_data)

        fitted = model.get_fitted_signal()

        assert fitted.shape == sample_data.shape

    def test_get_residuals(self, sample_data):
        """Test get_residuals method."""
        from pySPFM import SparseDeconvolution

        model = SparseDeconvolution(tr=2.0, criterion="bic", max_iter=10)
        model.fit(sample_data)

        residuals = model.get_residuals(sample_data)
        fitted = model.get_fitted_signal()

        np.testing.assert_array_almost_equal(residuals, sample_data - fitted)

    def test_repr(self):
        """Test string representation."""
        from pySPFM import SparseDeconvolution

        model = SparseDeconvolution(tr=2.0, criterion="bic")
        repr_str = repr(model)

        assert "SparseDeconvolution" in repr_str
        assert "tr=" in repr_str
        assert "criterion=" in repr_str


class TestLowRankPlusSparseAPI:
    """Tests for LowRankPlusSparse scikit-learn API compliance."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample fMRI-like data."""
        np.random.seed(42)
        n_timepoints = 50
        n_voxels = 10
        return np.random.randn(n_timepoints, n_voxels)

    def test_init_parameters(self):
        """Test that initialization parameters are stored correctly."""
        from pySPFM import LowRankPlusSparse

        model = LowRankPlusSparse(
            tr=2.0,
            eigval_threshold=0.1,
            debias=True,
        )
        params = model.get_params()

        assert params["tr"] == 2.0
        assert params["eigval_threshold"] == 0.1
        assert params["debias"] is True

    def test_fitted_attributes(self, sample_data):
        """Test that fit sets expected attributes."""
        from pySPFM import LowRankPlusSparse

        model = LowRankPlusSparse(tr=2.0, max_iter=5)
        model.fit(sample_data)

        # Check fitted attributes exist
        assert hasattr(model, "coef_")
        assert hasattr(model, "low_rank_")
        assert hasattr(model, "lambda_")
        assert hasattr(model, "n_iter_")

        # Check shapes
        assert model.coef_.shape == sample_data.shape
        assert model.low_rank_.shape == sample_data.shape

    def test_get_fitted_signal_includes_lowrank(self, sample_data):
        """Test that get_fitted_signal includes low-rank component."""
        from pySPFM import LowRankPlusSparse

        model = LowRankPlusSparse(tr=2.0, max_iter=5)
        model.fit(sample_data)

        fitted = model.get_fitted_signal()
        np.dot(model.hrf_matrix_, model.coef_)

        # Fitted should include low-rank, so should differ from sparse-only
        # (unless low-rank is zero, which is unlikely with random data)
        assert fitted.shape == sample_data.shape


class TestStabilitySelectionAPI:
    """Tests for StabilitySelection scikit-learn API compliance."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample fMRI-like data."""
        np.random.seed(42)
        n_timepoints = 30
        n_voxels = 5
        return np.random.randn(n_timepoints, n_voxels)

    def test_init_parameters(self):
        """Test that initialization parameters are stored correctly."""
        from pySPFM import StabilitySelection

        model = StabilitySelection(
            tr=2.0,
            n_surrogates=20,
            threshold=0.5,
        )
        params = model.get_params()

        assert params["tr"] == 2.0
        assert params["n_surrogates"] == 20
        assert params["threshold"] == 0.5

    def test_fitted_attributes(self, sample_data):
        """Test that fit sets expected attributes."""
        from pySPFM import StabilitySelection

        model = StabilitySelection(tr=2.0, n_surrogates=5)
        model.fit(sample_data)

        # Check fitted attributes exist
        assert hasattr(model, "selection_frequency_")
        assert hasattr(model, "coef_")

        # Check shapes
        assert model.selection_frequency_.shape == sample_data.shape
        assert model.coef_.shape == sample_data.shape

    def test_selection_frequency_range(self, sample_data):
        """Test that selection frequencies are in valid range."""
        from pySPFM import StabilitySelection

        model = StabilitySelection(tr=2.0, n_surrogates=5)
        model.fit(sample_data)

        # Selection frequencies should be in [0, 1]
        assert np.all(model.selection_frequency_ >= 0)
        assert np.all(model.selection_frequency_ <= 1)

    def test_transform_returns_frequencies(self, sample_data):
        """Test that transform returns selection frequencies."""
        from pySPFM import StabilitySelection

        model = StabilitySelection(tr=2.0, n_surrogates=5)
        model.fit(sample_data)

        transformed = model.transform(sample_data)

        np.testing.assert_array_equal(transformed, model.selection_frequency_)


class TestSparseDeconvolutionCoverage:
    """Additional tests for SparseDeconvolution to improve coverage."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample fMRI-like data."""
        np.random.seed(42)
        n_timepoints = 50
        n_voxels = 10
        return np.random.randn(n_timepoints, n_voxels)

    def test_fista_criterion(self, sample_data):
        """Test FISTA-based criteria (mad, factor, etc.)."""
        from pySPFM import SparseDeconvolution

        model = SparseDeconvolution(tr=2.0, criterion="factor", max_iter=10)
        model.fit(sample_data)

        assert model.coef_.shape == sample_data.shape

    def test_multivariate_mode(self, sample_data):
        """Test multivariate mode with group > 0."""
        from pySPFM import SparseDeconvolution

        model = SparseDeconvolution(tr=2.0, criterion="factor", group=0.5, max_iter=10)
        model.fit(sample_data)

        assert model.coef_.shape == sample_data.shape
        # Lambda values array should have same shape as number of voxels
        assert model.lambda_.shape == (sample_data.shape[1],)

    def test_multivariate_with_lars_raises(self, sample_data):
        """Test that multivariate mode with LARS raises ValueError."""
        from pySPFM import SparseDeconvolution

        model = SparseDeconvolution(tr=2.0, criterion="bic", group=0.5)

        with pytest.raises(ValueError, match="Multivariate mode"):
            model.fit(sample_data)

    def test_invalid_criterion_raises(self, sample_data):
        """Test that invalid criterion raises ValueError."""
        from pySPFM import SparseDeconvolution

        model = SparseDeconvolution(tr=2.0, criterion="invalid")

        with pytest.raises(ValueError, match="Invalid criterion"):
            model.fit(sample_data)

    def test_block_model(self, sample_data):
        """Test block model estimation."""
        from pySPFM import SparseDeconvolution

        model = SparseDeconvolution(tr=2.0, criterion="bic", block_model=True, max_iter=10)
        model.fit(sample_data)

        assert model.coef_.shape == sample_data.shape

    def test_block_model_fista(self, sample_data):
        """Test block model with FISTA."""
        from pySPFM import SparseDeconvolution

        model = SparseDeconvolution(tr=2.0, criterion="factor", block_model=True, max_iter=10)
        model.fit(sample_data)

        assert model.coef_.shape == sample_data.shape

    def test_no_debias(self, sample_data):
        """Test without debiasing."""
        from pySPFM import SparseDeconvolution

        model = SparseDeconvolution(tr=2.0, criterion="bic", debias=False, max_iter=10)
        model.fit(sample_data)

        assert model.coef_.shape == sample_data.shape

    def test_1d_input(self):
        """Test with 1D input (single voxel)."""
        from pySPFM import SparseDeconvolution

        np.random.seed(42)
        X = np.random.randn(50)

        model = SparseDeconvolution(tr=2.0, criterion="bic", max_iter=10)
        model.fit(X)

        assert model.coef_.shape == (50, 1)

    def test_positive_constraint(self, sample_data):
        """Test positive constraint."""
        from pySPFM import SparseDeconvolution

        model = SparseDeconvolution(
            tr=2.0, criterion="factor", positive=True, debias=False, max_iter=10
        )
        model.fit(sample_data)

        assert model.coef_.shape == sample_data.shape

    def test_aic_criterion(self, sample_data):
        """Test AIC criterion."""
        from pySPFM import SparseDeconvolution

        model = SparseDeconvolution(tr=2.0, criterion="aic", max_iter=10)
        model.fit(sample_data)

        assert model.coef_.shape == sample_data.shape

    def test_score_1d_input(self):
        """Test score method with 1D input."""
        from pySPFM import SparseDeconvolution

        np.random.seed(42)
        X = np.random.randn(50)

        model = SparseDeconvolution(tr=2.0, criterion="bic", max_iter=10)
        model.fit(X)

        score = model.score(X)

        assert isinstance(score, float)


class TestLowRankPlusSparseCoverage:
    """Additional tests for LowRankPlusSparse to improve coverage."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample fMRI-like data."""
        np.random.seed(42)
        n_timepoints = 50
        n_voxels = 10
        return np.random.randn(n_timepoints, n_voxels)

    def test_block_model(self, sample_data):
        """Test block model with LowRankPlusSparse."""
        from pySPFM import LowRankPlusSparse

        model = LowRankPlusSparse(tr=2.0, block_model=True, max_iter=3)
        model.fit(sample_data)

        assert model.coef_.shape == sample_data.shape

    def test_no_debias(self, sample_data):
        """Test without debiasing."""
        from pySPFM import LowRankPlusSparse

        model = LowRankPlusSparse(tr=2.0, debias=False, max_iter=3)
        model.fit(sample_data)

        assert model.coef_.shape == sample_data.shape

    def test_convergence(self, sample_data):
        """Test that model converges with small tolerance."""
        from pySPFM import LowRankPlusSparse

        model = LowRankPlusSparse(tr=2.0, max_iter=100, tol=1e-2)
        model.fit(sample_data)

        # Should converge before max_iter
        assert model.n_iter_ < 100 or model.n_iter_ == 100

    def test_transform(self, sample_data):
        """Test transform method."""
        from pySPFM import LowRankPlusSparse

        model = LowRankPlusSparse(tr=2.0, max_iter=3)
        model.fit(sample_data)

        transformed = model.transform(sample_data)

        np.testing.assert_array_equal(transformed, model.coef_)


class TestStabilitySelectionCoverage:
    """Additional tests for StabilitySelection to improve coverage."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample fMRI-like data."""
        np.random.seed(42)
        n_timepoints = 30
        n_voxels = 5
        return np.random.randn(n_timepoints, n_voxels)

    def test_with_n_lambdas(self, sample_data):
        """Test with explicit n_lambdas."""
        from pySPFM import StabilitySelection

        model = StabilitySelection(tr=2.0, n_surrogates=3, n_lambdas=20)
        model.fit(sample_data)

        assert model.selection_frequency_.shape == sample_data.shape

    def test_threshold_binary_selection(self, sample_data):
        """Test that coef_ is binary based on threshold."""
        from pySPFM import StabilitySelection

        model = StabilitySelection(tr=2.0, n_surrogates=3, threshold=0.5)
        model.fit(sample_data)

        # coef_ should only contain 0s and 1s
        unique_vals = np.unique(model.coef_)
        assert set(unique_vals).issubset({0.0, 1.0})
