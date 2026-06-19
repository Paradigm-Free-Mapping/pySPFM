import numpy as np
import pytest

from pySPFM._solvers.fista import fista


def test_fista(sim_data, sim_hrf, fista_results):
    y = np.load(sim_data, allow_pickle=True)
    hrf_matrix = np.load(sim_hrf, allow_pickle=True)

    # Run FISTA without pylops
    estimates, lambda_ = fista(hrf_matrix, y, group=0.2, use_pylops=False, max_iter=50)

    # Compare estimates
    assert np.allclose(lambda_, np.repeat(np.array([0.60157822]), y.shape[1], axis=0))
    assert np.allclose(estimates, np.load(fista_results, allow_pickle=True), atol=1e-6)


def test_fista_positives(sim_data, sim_hrf, fista_positives):
    y = np.load(sim_data, allow_pickle=True)
    hrf_matrix = np.load(sim_hrf, allow_pickle=True)

    # Run FISTA without pylops
    estimates, lambda_ = fista(
        hrf_matrix, y, group=0.2, use_pylops=False, max_iter=50, positive_only=True
    )

    # Compare estimates. The positive_only clamp is sensitive to small numerical
    # differences across jax/jaxlib versions (CI installs the latest, unpinned),
    # so use a looser tolerance here than the other reference comparisons.
    assert np.allclose(lambda_, np.repeat(np.array([0.60157822]), y.shape[1], axis=0))
    assert np.allclose(
        estimates, np.maximum(np.load(fista_positives, allow_pickle=True), 0), atol=1e-4
    )


def test_fista_pylops(sim_data, sim_hrf, pylops_results):
    y = np.load(sim_data, allow_pickle=True)
    hrf_matrix = np.load(sim_hrf, allow_pickle=True)

    # Run FISTA with pylops
    estimates, lambda_ = fista(hrf_matrix, y, group=0.2, use_pylops=True, max_iter=50)

    # Compare estimates
    assert np.allclose(lambda_, np.repeat(np.array([0.60157822]), y.shape[1], axis=0))
    assert np.allclose(estimates, np.load(pylops_results, allow_pickle=True), atol=1e-6)


def test_fista_weights_neutral(sim_data, sim_hrf):
    """Unit weights (w=1) reproduce the unweighted multivariate result."""
    y = np.load(sim_data, allow_pickle=True)
    hrf_matrix = np.load(sim_hrf, allow_pickle=True)
    n_voxels = y.shape[1]

    est_none, lambda_none = fista(hrf_matrix, y, group=0.2, use_pylops=False, max_iter=50)
    est_ones, lambda_ones = fista(
        hrf_matrix, y, group=0.2, use_pylops=False, max_iter=50, weights=np.ones(n_voxels)
    )

    assert np.allclose(lambda_none, lambda_ones)
    assert np.allclose(est_none, est_ones, atol=1e-6)


def test_fista_weights_direction(sim_data, sim_hrf):
    """Higher weights penalize less (retain more activity); lower weights penalize more."""
    y = np.load(sim_data, allow_pickle=True)
    hrf_matrix = np.load(sim_hrf, allow_pickle=True)
    n_voxels = y.shape[1]

    est_neutral, _ = fista(hrf_matrix, y, group=0.2, use_pylops=False, max_iter=50)
    est_less_penalty, _ = fista(
        hrf_matrix, y, group=0.2, use_pylops=False, max_iter=50, weights=np.full(n_voxels, 2.0)
    )
    est_more_penalty, _ = fista(
        hrf_matrix, y, group=0.2, use_pylops=False, max_iter=50, weights=np.full(n_voxels, 0.5)
    )

    # Total L1 energy of the estimates is monotonic in the penalty strength.
    assert np.sum(np.abs(est_less_penalty)) > np.sum(np.abs(est_neutral))
    assert np.sum(np.abs(est_neutral)) > np.sum(np.abs(est_more_penalty))


def test_fista_weights_validation(sim_data, sim_hrf):
    """Invalid weights raise informative errors."""
    y = np.load(sim_data, allow_pickle=True)
    hrf_matrix = np.load(sim_hrf, allow_pickle=True)
    n_voxels = y.shape[1]

    # Wrong shape
    with pytest.raises(ValueError, match="weights must have shape"):
        fista(hrf_matrix, y, group=0.2, max_iter=5, weights=np.ones(n_voxels + 1))

    # Non-positive values
    bad = np.ones(n_voxels)
    bad[0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        fista(hrf_matrix, y, group=0.2, max_iter=5, weights=bad)

    # Non-finite values (NaN / Inf)
    for bad_val in (np.nan, np.inf):
        bad_nf = np.ones(n_voxels)
        bad_nf[0] = bad_val
        with pytest.raises(ValueError, match="finite and strictly positive"):
            fista(hrf_matrix, y, group=0.2, max_iter=5, weights=bad_nf)

    # Weights require multivariate mode (group > 0)
    with pytest.raises(ValueError, match="multivariate mode"):
        fista(hrf_matrix, y, group=0.0, max_iter=5, weights=np.ones(n_voxels))

    # Weights are not supported with the pylops backend
    with pytest.raises(ValueError, match="pylops"):
        fista(hrf_matrix, y, group=0.2, max_iter=5, use_pylops=True, weights=np.ones(n_voxels))
