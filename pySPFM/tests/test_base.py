"""Tests for base classes and scikit-learn API compliance."""

import numpy as np
import pytest

from pySPFM.base import (
    BaseEstimator,
    NotFittedError,
    TransformerMixin,
    check_is_fitted,
    clone,
)


class SimpleEstimator(BaseEstimator):
    """Simple test estimator."""

    def __init__(self, *, param1=1, param2="default"):
        self.param1 = param1
        self.param2 = param2

    def fit(self, X, y=None):
        self.fitted_ = True
        self.coef_ = np.ones(X.shape[1])
        return self


class SimpleTransformer(TransformerMixin, BaseEstimator):
    """Simple test transformer."""

    def __init__(self, *, scale=1.0):
        self.scale = scale

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        return X * self.scale


class TestBaseEstimator:
    """Tests for BaseEstimator class."""

    def test_get_params(self):
        """Test that get_params returns correct parameters."""
        est = SimpleEstimator(param1=5, param2="test")
        params = est.get_params()
        assert params == {"param1": 5, "param2": "test"}

    def test_set_params(self):
        """Test that set_params sets parameters correctly."""
        est = SimpleEstimator()
        est.set_params(param1=10, param2="new")
        assert est.param1 == 10
        assert est.param2 == "new"

    def test_set_params_invalid(self):
        """Test that set_params raises error for invalid parameters."""
        est = SimpleEstimator()
        with pytest.raises(ValueError, match="Invalid parameter"):
            est.set_params(invalid_param=1)

    def test_repr(self):
        """Test string representation."""
        est = SimpleEstimator(param1=5)
        repr_str = repr(est)
        assert "SimpleEstimator" in repr_str
        assert "param1=5" in repr_str


class TestClone:
    """Tests for clone function."""

    def test_clone_unfitted(self):
        """Test cloning an unfitted estimator."""
        est = SimpleEstimator(param1=5)
        cloned = clone(est)
        assert cloned is not est
        assert cloned.get_params() == est.get_params()

    def test_clone_fitted(self):
        """Test that clone does not copy fitted attributes."""
        X = np.random.randn(10, 5)
        est = SimpleEstimator(param1=5)
        est.fit(X)
        assert hasattr(est, "fitted_")

        cloned = clone(est)
        assert not hasattr(cloned, "fitted_")
        assert cloned.get_params() == est.get_params()


class TestCheckIsFitted:
    """Tests for check_is_fitted function."""

    def test_unfitted_estimator(self):
        """Test that unfitted estimator raises NotFittedError."""
        est = SimpleEstimator()
        with pytest.raises(NotFittedError):
            check_is_fitted(est)

    def test_fitted_estimator(self):
        """Test that fitted estimator passes check."""
        X = np.random.randn(10, 5)
        est = SimpleEstimator()
        est.fit(X)
        # Should not raise
        check_is_fitted(est)

    def test_specific_attributes(self):
        """Test checking for specific attributes."""
        X = np.random.randn(10, 5)
        est = SimpleEstimator()
        est.fit(X)

        # Should pass with correct attribute
        check_is_fitted(est, ["coef_"])

        # Should fail with wrong attribute
        with pytest.raises(NotFittedError):
            check_is_fitted(est, ["wrong_attr_"])


class TestTransformerMixin:
    """Tests for TransformerMixin class."""

    def test_fit_transform(self):
        """Test that fit_transform works correctly."""
        X = np.random.randn(10, 5)
        transformer = SimpleTransformer(scale=2.0)

        X_transformed = transformer.fit_transform(X)

        assert transformer.fitted_
        np.testing.assert_array_almost_equal(X_transformed, X * 2.0)

    def test_fit_then_transform(self):
        """Test that fit then transform gives same result as fit_transform."""
        X = np.random.randn(10, 5)
        transformer1 = SimpleTransformer(scale=2.0)
        transformer2 = SimpleTransformer(scale=2.0)

        X1 = transformer1.fit_transform(X)
        X2 = transformer2.fit(X).transform(X)

        np.testing.assert_array_almost_equal(X1, X2)


class TestNotFittedError:
    """Tests for NotFittedError exception."""

    def test_inheritance(self):
        """Test that NotFittedError inherits from correct classes."""
        assert issubclass(NotFittedError, ValueError)
        assert issubclass(NotFittedError, AttributeError)

    def test_exception_message(self):
        """Test exception message formatting."""
        try:
            raise NotFittedError("Test message")
        except NotFittedError as e:
            assert "Test message" in str(e)


class TestCheckIsFittedEdgeCases:
    """Additional tests for check_is_fitted to improve coverage."""

    def test_not_an_estimator(self):
        """Test that non-estimator raises TypeError."""
        not_estimator = "just a string"
        with pytest.raises(TypeError, match="is not an estimator"):
            check_is_fitted(not_estimator)

    def test_custom_message(self):
        """Test custom error message."""
        est = SimpleEstimator()
        custom_msg = "Custom message for %(name)s"
        with pytest.raises(NotFittedError, match="Custom message for SimpleEstimator"):
            check_is_fitted(est, msg=custom_msg)

    def test_all_or_any_parameter(self):
        """Test all_or_any parameter."""
        X = np.random.randn(10, 5)
        est = SimpleEstimator()
        est.fit(X)

        # Should pass with any=any (one attribute exists)
        check_is_fitted(est, ["coef_", "nonexistent_"], all_or_any=any)

        # Should fail with all=all (not all attributes exist)
        with pytest.raises(NotFittedError):
            check_is_fitted(est, ["coef_", "nonexistent_"], all_or_any=all)


class TestBaseEstimatorValidateParams:
    """Tests for _validate_params method."""

    def test_validate_params_no_constraints(self):
        """Test that validation passes when no constraints defined."""
        est = SimpleEstimator()
        # Should not raise - no constraints defined
        est._validate_params()

    def test_validate_params_with_constraints(self):
        """Test validation with constraints."""
        from pySPFM import SparseDeconvolution

        # SparseDeconvolution has _parameter_constraints defined
        model = SparseDeconvolution(tr=2.0)
        # Should not raise with valid params
        model._validate_params()


class TestDeconvolutionMixin:
    """Tests for DeconvolutionMixin to improve coverage."""

    @pytest.fixture
    def fitted_model(self):
        """Create a fitted SparseDeconvolution model."""
        from pySPFM import SparseDeconvolution

        np.random.seed(42)
        X = np.random.randn(50, 10)
        model = SparseDeconvolution(tr=2.0, criterion="bic", max_iter=10)
        model.fit(X)
        return model, X

    def test_score_with_zero_variance(self, fitted_model):
        """Test score when some voxels have zero variance."""
        model, X = fitted_model

        # Create data with zero variance in one column
        X_zero_var = X.copy()
        X_zero_var[:, 0] = 5.0  # Constant column

        score = model.score(X_zero_var)

        # Should still return a valid float
        assert isinstance(score, float)

    def test_get_residuals_1d(self, fitted_model):
        """Test get_residuals with 1D input."""
        model, _ = fitted_model

        # Create 1D input
        X_1d = np.random.randn(50)

        # This should handle 1D input
        residuals = model.get_residuals(X_1d)

        assert residuals.ndim == 2


class TestValidateData:
    """Tests for _validate_data function."""

    def test_validate_data_basic(self):
        """Test basic _validate_data functionality."""
        from pySPFM.base import _validate_data

        est = SimpleEstimator()
        X = np.random.randn(10, 5)

        X_validated = _validate_data(est, X)

        assert X_validated.shape == X.shape
        assert est.n_features_in_ == 5
        assert est.n_samples_ == 10

    def test_validate_data_1d(self):
        """Test _validate_data with 1D input."""
        from pySPFM.base import _validate_data

        est = SimpleEstimator()
        X = np.random.randn(10)

        X_validated = _validate_data(est, X)

        assert X_validated.shape == (10, 1)

    def test_validate_data_with_y(self):
        """Test _validate_data with y input."""
        from pySPFM.base import _validate_data

        est = SimpleEstimator()
        X = np.random.randn(10, 5)
        y = np.random.randn(10)

        X_validated, y_validated = _validate_data(est, X, y)

        assert X_validated.shape == X.shape
        assert y_validated.shape == y.shape

    def test_validate_data_no_reset(self):
        """Test _validate_data without resetting attributes."""
        from pySPFM.base import _validate_data

        est = SimpleEstimator()
        est.n_features_in_ = 100  # Set existing value

        X = np.random.randn(10, 5)
        _validate_data(est, X, reset=False)

        # Should keep the original value when reset=False
        assert est.n_features_in_ == 100
