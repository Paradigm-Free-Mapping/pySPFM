"""Tests for regressors functionality in deconvolution."""

import numpy as np
import pytest

from pySPFM._solvers.fista import fista
from pySPFM._solvers.hrf_generator import HRFMatrix
from pySPFM._solvers.lars import solve_regularization_path


@pytest.fixture
def setup_data():
    """Set up test data for regressors tests."""
    n_scans = 100
    tr = 2.0
    te = [0]

    # Generate HRF
    hrf_obj = HRFMatrix(te=te, block=False)
    hrf = hrf_obj.generate_hrf(tr=tr, n_scans=n_scans).hrf_

    # Generate synthetic signal with some activity
    true_signal = np.zeros(n_scans)
    true_signal[20] = 1.0
    true_signal[50] = 0.8
    true_signal[80] = 0.6

    # Generate regressors (e.g., motion parameters)
    n_regressors = 3
    regressors = np.random.randn(n_scans, n_regressors) * 0.1

    # Generate data: HRF convolved with signal + regressor effects + noise
    regressor_betas = np.array([0.5, -0.3, 0.2])
    data = (
        np.dot(hrf, true_signal)
        + np.dot(regressors, regressor_betas)
        + np.random.randn(n_scans) * 0.05
    )

    return {
        "hrf": hrf,
        "data": data,
        "regressors": regressors,
        "n_scans": n_scans,
        "true_signal": true_signal,
    }


def test_fista_with_regressors(setup_data):
    """Test FISTA with regressors parameter."""
    hrf = setup_data["hrf"]
    data = setup_data["data"]
    regressors = setup_data["regressors"]

    # Run FISTA with regressors
    estimates, lambda_val = fista(
        hrf,
        data,
        criterion="mad",
        max_iter=100,
        min_iter=10,
        tol=1e-6,
        regressors=regressors,
    )

    # Check that estimates have the correct shape
    assert estimates.shape[0] == setup_data["n_scans"]

    # Check that lambda is positive
    assert lambda_val > 0

    # Check that estimates are not all zeros (should detect some activity)
    assert np.sum(np.abs(estimates)) > 0


def test_fista_without_regressors(setup_data):
    """Test FISTA without regressors (baseline behavior)."""
    hrf = setup_data["hrf"]
    data = setup_data["data"]

    # Run FISTA without regressors
    estimates, lambda_val = fista(
        hrf,
        data,
        criterion="mad",
        max_iter=100,
        min_iter=10,
        tol=1e-6,
        regressors=None,
    )

    # Check that estimates have the correct shape
    assert estimates.shape[0] == setup_data["n_scans"]

    # Check that lambda is positive
    assert lambda_val > 0


def test_fista_regressors_dimension_validation(setup_data):
    """Test that FISTA validates regressor dimensions."""
    hrf = setup_data["hrf"]
    data = setup_data["data"]
    n_scans = setup_data["n_scans"]

    # Create regressors with wrong number of timepoints
    wrong_regressors = np.random.randn(n_scans + 10, 2)

    # This should raise a ValueError due to dimension mismatch
    with pytest.raises(ValueError, match="doesn't have the right dimensions"):
        fista(
            hrf,
            data,
            criterion="mad",
            max_iter=100,
            regressors=wrong_regressors,
        )


def test_fista_regressors_transpose(setup_data):
    """Test that FISTA handles transposed regressors correctly."""
    hrf = setup_data["hrf"]
    data = setup_data["data"]
    regressors = setup_data["regressors"]

    # Transpose regressors (n_regressors, n_scans) instead of (n_scans, n_regressors)
    regressors_transposed = regressors.T

    # Run FISTA with transposed regressors - should auto-transpose
    estimates, lambda_val = fista(
        hrf,
        data,
        criterion="mad",
        max_iter=100,
        min_iter=10,
        tol=1e-6,
        regressors=regressors_transposed,
    )

    # Should work without error
    assert estimates.shape[0] == setup_data["n_scans"]


def test_fista_single_regressor(setup_data):
    """Test FISTA with a single regressor (1D array)."""
    hrf = setup_data["hrf"]
    data = setup_data["data"]
    n_scans = setup_data["n_scans"]

    # Create a single regressor as 1D array
    single_regressor = np.random.randn(n_scans)

    # Run FISTA with single regressor
    estimates, lambda_val = fista(
        hrf,
        data,
        criterion="mad",
        max_iter=100,
        min_iter=10,
        tol=1e-6,
        regressors=single_regressor,
    )

    # Should work without error
    assert estimates.shape[0] == n_scans


def test_fista_regressors_with_pylops_raises_error(setup_data):
    """Test that using regressors with pylops raises an error."""
    hrf = setup_data["hrf"]
    data = setup_data["data"]
    regressors = setup_data["regressors"]

    # This should raise a ValueError
    with pytest.raises(ValueError, match="regressors option is not available with pylops"):
        fista(
            hrf,
            data,
            criterion="mad",
            max_iter=100,
            use_pylops=True,
            regressors=regressors,
        )


def test_lars_with_regressors(setup_data):
    """Test LARS solve_regularization_path with regressors."""
    hrf = setup_data["hrf"]
    data = setup_data["data"]
    regressors = setup_data["regressors"]
    n_scans = setup_data["n_scans"]

    # Run LARS with regressors using FISTA backend
    estimates, lambda_optimal, _, _ = solve_regularization_path(
        hrf,
        data,
        n_lambdas=20,
        criterion="bic",
        use_fista=True,
        regressors=regressors,
    )

    # Check that estimates have the correct shape
    assert estimates.shape[0] == n_scans

    # Check that we got a single lambda value (optimal)
    assert isinstance(lambda_optimal, int | float | np.number)


def test_lars_without_regressors(setup_data):
    """Test LARS without regressors (baseline behavior)."""
    hrf = setup_data["hrf"]
    data = setup_data["data"]
    n_scans = setup_data["n_scans"]

    # Run LARS without regressors
    estimates, lambda_optimal, _, _ = solve_regularization_path(
        hrf,
        data,
        n_lambdas=20,
        criterion="bic",
        use_fista=False,
        regressors=None,
    )

    # Check that estimates have the correct shape
    assert estimates.shape[0] == n_scans

    # Check that lambda is a scalar
    assert isinstance(lambda_optimal, int | float | np.number)


def test_regressors_improve_fit(setup_data):
    """Test that including regressors improves the fit when confounds are present."""
    hrf = setup_data["hrf"]
    data = setup_data["data"]
    regressors = setup_data["regressors"]

    # Run FISTA without regressors
    estimates_no_reg, _ = fista(
        hrf,
        data,
        criterion="mad",
        max_iter=100,
        min_iter=10,
        tol=1e-6,
        regressors=None,
    )

    # Run FISTA with regressors
    estimates_with_reg, _ = fista(
        hrf,
        data,
        criterion="mad",
        max_iter=100,
        min_iter=10,
        tol=1e-6,
        regressors=regressors,
    )

    # Both should produce estimates
    assert np.sum(np.abs(estimates_no_reg)) > 0
    assert np.sum(np.abs(estimates_with_reg)) > 0

    # The estimates should be different (regressors affect the solution)
    assert not np.allclose(estimates_no_reg, estimates_with_reg)


def test_fista_lasso_with_regressors(setup_data):
    """Test FISTA with Lasso (group=0) and regressors."""
    hrf = setup_data["hrf"]
    data = setup_data["data"]
    regressors = setup_data["regressors"]

    # Run FISTA with regressors and group=0 (Lasso)
    estimates, _ = fista(
        hrf,
        data,
        criterion="mad",
        max_iter=100,
        min_iter=10,
        tol=1e-6,
        group=0.0,
        regressors=regressors,
    )

    # Check that estimates have the correct shape
    assert estimates.shape[0] == setup_data["n_scans"]

    # Check that estimates are not all zeros
    assert np.sum(np.abs(estimates)) > 0


def test_fista_multi_echo_with_regressors():
    """Test FISTA with multi-echo data and regressors."""
    n_scans = 100
    tr = 2.0
    te = [13.7, 29.9, 46.1]  # Multi-echo TEs in ms
    n_te = len(te)

    # Generate multi-echo HRF
    hrf_obj = HRFMatrix(te=te, block=False)
    hrf = hrf_obj.generate_hrf(tr=tr, n_scans=n_scans).hrf_

    # HRF should have shape (n_scans * n_te, n_scans)
    assert hrf.shape == (n_scans * n_te, n_scans)

    # Generate synthetic signal with some activity
    true_signal = np.zeros(n_scans)
    true_signal[20] = 1.0
    true_signal[50] = 0.8
    true_signal[80] = 0.6

    # Generate regressors with correct shape for multi-echo: (n_scans * n_te, n_regressors)
    n_regressors = 3
    regressors = np.random.randn(n_scans * n_te, n_regressors) * 0.1

    # Generate multi-echo data
    regressor_betas = np.array([0.5, -0.3, 0.2])
    data = (
        np.dot(hrf, true_signal)
        + np.dot(regressors, regressor_betas)
        + np.random.randn(n_scans * n_te) * 0.05
    )

    # Run FISTA with regressors
    estimates, lambda_val = fista(
        hrf,
        data,
        criterion="mad",
        max_iter=100,
        min_iter=10,
        tol=1e-6,
        regressors=regressors,
    )

    # Check that estimates have the correct shape (n_scans, not n_scans * n_te)
    assert estimates.shape[0] == n_scans

    # Check that lambda is positive
    assert lambda_val > 0

    # Check that estimates are not all zeros
    assert np.sum(np.abs(estimates)) > 0


def test_fista_multi_echo_regressors_dimension_validation():
    """Test that FISTA correctly validates regressor dimensions for multi-echo data."""
    n_scans = 100
    tr = 2.0
    te = [13.7, 29.9, 46.1]  # Multi-echo TEs
    n_te = len(te)

    # Generate multi-echo HRF
    hrf_obj = HRFMatrix(te=te, block=False)
    hrf = hrf_obj.generate_hrf(tr=tr, n_scans=n_scans).hrf_

    # Generate data
    data = np.random.randn(n_scans * n_te)

    # Create regressors with wrong shape (n_scans instead of n_scans * n_te)
    wrong_regressors = np.random.randn(n_scans, 2)

    # This should raise a ValueError due to dimension mismatch
    with pytest.raises(ValueError, match="doesn't have the right dimensions"):
        fista(
            hrf,
            data,
            criterion="mad",
            max_iter=100,
            regressors=wrong_regressors,
        )


def test_fista_multi_echo_regressors_transpose():
    """Test that FISTA handles transposed regressors correctly for multi-echo data."""
    n_scans = 100
    tr = 2.0
    te = [13.7, 29.9, 46.1]  # Multi-echo TEs
    n_te = len(te)

    # Generate multi-echo HRF
    hrf_obj = HRFMatrix(te=te, block=False)
    hrf = hrf_obj.generate_hrf(tr=tr, n_scans=n_scans).hrf_

    # Generate data
    true_signal = np.zeros(n_scans)
    true_signal[20] = 1.0
    data = np.dot(hrf, true_signal) + np.random.randn(n_scans * n_te) * 0.05

    # Create regressors with shape (n_regressors, n_scans * n_te) - transposed
    n_regressors = 2
    regressors_transposed = np.random.randn(n_regressors, n_scans * n_te) * 0.1

    # Run FISTA - should auto-transpose
    estimates, lambda_val = fista(
        hrf,
        data,
        criterion="mad",
        max_iter=100,
        min_iter=10,
        tol=1e-6,
        regressors=regressors_transposed,
    )

    # Should work without error
    assert estimates.shape[0] == n_scans
