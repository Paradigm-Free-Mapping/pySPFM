"""Tests for base classes and scikit-learn API compliance."""

import numpy as np
import pytest

from pySPFM.base import (
    BaseEstimator,
    DeconvolutionMixin,
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
