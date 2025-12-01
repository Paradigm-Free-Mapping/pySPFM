"""Tests for _utils module to improve coverage."""

import numpy as np
import pytest

from pySPFM._utils import (
    _estimator_repr,
    check_array,
    r2_score,
    validate_parameter_constraints,
)


class TestEstimatorRepr:
    """Tests for _estimator_repr function."""

    def test_repr_with_string_param(self):
        """Test repr with string parameters."""
        from pySPFM import SparseDeconvolution

        model = SparseDeconvolution(tr=2.0, criterion="bic")
        repr_str = _estimator_repr(model)

        assert "SparseDeconvolution" in repr_str
        assert "criterion='bic'" in repr_str

    def test_repr_with_float_param(self):
        """Test repr with float parameters."""
        from pySPFM import SparseDeconvolution

        model = SparseDeconvolution(tr=2.5)
        repr_str = _estimator_repr(model)

        assert "tr=2.5" in repr_str

    def test_repr_with_long_list_param(self):
        """Test repr with long list parameters (>3 elements)."""
        from pySPFM import SparseDeconvolution

        model = SparseDeconvolution(tr=2.0, te=[14.0, 28.0, 42.0, 56.0])
        repr_str = _estimator_repr(model)

        # Should truncate long lists
        assert "SparseDeconvolution" in repr_str

    def test_repr_truncation(self):
        """Test that very long repr is truncated."""
        from pySPFM import SparseDeconvolution

        model = SparseDeconvolution(tr=2.0)
        # Use a short max chars to force truncation
        repr_str = _estimator_repr(model, N_CHAR_MAX=50)

        assert len(repr_str) <= 50
        assert repr_str.endswith("...")


class TestR2Score:
    """Tests for r2_score function."""

    def test_r2_score_perfect(self):
        """Test R² score for perfect prediction."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        score = r2_score(y_true, y_pred)

        assert score == 1.0

    def test_r2_score_zero(self):
        """Test R² score when prediction equals mean."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([3, 3, 3, 3, 3])  # Mean of y_true

        score = r2_score(y_true, y_pred)

        assert np.isclose(score, 0.0)

    def test_r2_score_multioutput_raw(self):
        """Test R² score with raw_values multioutput."""
        y_true = np.array([[1, 2], [2, 3], [3, 4]])
        y_pred = np.array([[1, 2], [2, 3], [3, 4]])

        scores = r2_score(y_true, y_pred, multioutput="raw_values")

        assert isinstance(scores, np.ndarray)
        assert len(scores) == 2
        np.testing.assert_array_equal(scores, [1.0, 1.0])

    def test_r2_score_multioutput_uniform_average(self):
        """Test R² score with uniform_average multioutput."""
        y_true = np.array([[1, 2], [2, 3], [3, 4]])
        y_pred = np.array([[1, 2], [2, 3], [3, 4]])

        score = r2_score(y_true, y_pred, multioutput="uniform_average")

        assert isinstance(score, float)
        assert score == 1.0

    def test_r2_score_multioutput_variance_weighted(self):
        """Test R² score with variance_weighted multioutput."""
        y_true = np.array([[1, 1], [2, 5], [3, 9]])
        y_pred = np.array([[1, 1], [2, 5], [3, 9]])

        score = r2_score(y_true, y_pred, multioutput="variance_weighted")

        assert isinstance(score, float)
        assert score == 1.0

    def test_r2_score_with_sample_weight(self):
        """Test R² score with sample weights."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        weights = np.array([1, 1, 1, 1, 10])  # Weight last sample more

        score_weighted = r2_score(y_true, y_pred, sample_weight=weights)
        score_unweighted = r2_score(y_true, y_pred)

        # Both should be high but different
        assert 0.9 < score_weighted < 1.0
        assert 0.9 < score_unweighted < 1.0

    def test_r2_score_variance_weighted_with_sample_weight(self):
        """Test variance_weighted with sample weights."""
        y_true = np.array([[1, 2], [2, 3], [3, 4]])
        y_pred = np.array([[1, 2], [2, 3], [3, 4]])
        weights = np.array([1, 2, 3])

        score = r2_score(y_true, y_pred, sample_weight=weights, multioutput="variance_weighted")

        assert score == 1.0

    def test_r2_score_invalid_multioutput(self):
        """Test R² score with invalid multioutput raises error."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])

        with pytest.raises(ValueError, match="Invalid multioutput"):
            r2_score(y_true, y_pred, multioutput="invalid")

    def test_r2_score_zero_variance(self):
        """Test R² score when y_true has zero variance."""
        y_true = np.array([5, 5, 5, 5, 5])  # Zero variance
        y_pred = np.array([1, 2, 3, 4, 5])

        score = r2_score(y_true, y_pred)

        # Should return 0 when ss_tot is 0
        assert score == 0.0


class TestCheckArray:
    """Tests for check_array function."""

    def test_check_array_1d_to_2d(self):
        """Test 1D array is converted to 2D."""
        arr = np.array([1, 2, 3])
        result = check_array(arr)

        assert result.ndim == 2
        assert result.shape == (3, 1)

    def test_check_array_2d_unchanged(self):
        """Test 2D array is unchanged."""
        arr = np.array([[1, 2], [3, 4]])
        result = check_array(arr)

        assert result.ndim == 2
        np.testing.assert_array_equal(result, arr)

    def test_check_array_copy(self):
        """Test copy parameter."""
        arr = np.array([[1, 2], [3, 4]])
        result = check_array(arr, copy=True)

        assert result is not arr
        np.testing.assert_array_equal(result, arr)

    def test_check_array_dtype_conversion(self):
        """Test dtype conversion."""
        arr = np.array([1, 2, 3], dtype=np.int32)
        result = check_array(arr, dtype=np.float64)

        assert result.dtype == np.float64

    def test_check_array_disallow_nd(self):
        """Test that 3D array raises error when allow_nd=False."""
        arr = np.ones((2, 3, 4))

        with pytest.raises(ValueError, match="Expected <= 2"):
            check_array(arr, ensure_2d=False, allow_nd=False)

    def test_check_array_allow_nd(self):
        """Test 3D array is allowed when allow_nd=True."""
        arr = np.ones((2, 3, 4))
        result = check_array(arr, ensure_2d=False, allow_nd=True)

        assert result.shape == (2, 3, 4)


class TestValidateParameterConstraints:
    """Tests for validate_parameter_constraints function."""

    def test_validate_type_constraint(self):
        """Test type constraint validation."""
        constraints = {"param1": [int]}
        params = {"param1": 5}

        # Should not raise
        validate_parameter_constraints(constraints, params, "TestClass")

    def test_validate_type_constraint_invalid(self):
        """Test type constraint fails for wrong type."""
        constraints = {"param1": [int]}
        params = {"param1": "string"}

        with pytest.raises(ValueError, match="must satisfy the constraints"):
            validate_parameter_constraints(constraints, params, "TestClass")

    def test_validate_interval_constraint_both(self):
        """Test interval constraint with closed='both'."""
        constraints = {"param1": [("interval", float, 0.0, 1.0, "both")]}
        params = {"param1": 0.5}

        # Should not raise
        validate_parameter_constraints(constraints, params, "TestClass")

        # Edge cases should work too
        params = {"param1": 0.0}
        validate_parameter_constraints(constraints, params, "TestClass")

        params = {"param1": 1.0}
        validate_parameter_constraints(constraints, params, "TestClass")

    def test_validate_interval_constraint_left(self):
        """Test interval constraint with closed='left'."""
        constraints = {"param1": [("interval", float, 0.0, 1.0, "left")]}

        # 0.0 should be valid (left closed)
        params = {"param1": 0.0}
        validate_parameter_constraints(constraints, params, "TestClass")

        # 1.0 should be invalid (right open)
        params = {"param1": 1.0}
        with pytest.raises(ValueError):
            validate_parameter_constraints(constraints, params, "TestClass")

    def test_validate_interval_constraint_right(self):
        """Test interval constraint with closed='right'."""
        constraints = {"param1": [("interval", float, 0.0, 1.0, "right")]}

        # 0.0 should be invalid (left open)
        params = {"param1": 0.0}
        with pytest.raises(ValueError):
            validate_parameter_constraints(constraints, params, "TestClass")

        # 1.0 should be valid (right closed)
        params = {"param1": 1.0}
        validate_parameter_constraints(constraints, params, "TestClass")

    def test_validate_interval_constraint_neither(self):
        """Test interval constraint with closed='neither'."""
        constraints = {"param1": [("interval", float, 0.0, 1.0, "neither")]}

        # Edge values should be invalid
        params = {"param1": 0.0}
        with pytest.raises(ValueError):
            validate_parameter_constraints(constraints, params, "TestClass")

        params = {"param1": 1.0}
        with pytest.raises(ValueError):
            validate_parameter_constraints(constraints, params, "TestClass")

        # Middle value should be valid
        params = {"param1": 0.5}
        validate_parameter_constraints(constraints, params, "TestClass")

    def test_validate_callable_constraint(self):
        """Test callable constraint validation."""
        constraints = {"param1": [lambda x: x > 0]}
        params = {"param1": 5}

        # Should not raise
        validate_parameter_constraints(constraints, params, "TestClass")

        params = {"param1": -5}
        with pytest.raises(ValueError):
            validate_parameter_constraints(constraints, params, "TestClass")

    def test_validate_missing_param(self):
        """Test that missing params are ignored."""
        constraints = {"param1": [int], "param2": [str]}
        params = {"param1": 5}  # param2 is missing

        # Should not raise - missing params are skipped
        validate_parameter_constraints(constraints, params, "TestClass")
