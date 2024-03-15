import numpy as np

from pySPFM.deconvolution import fista


def test_fista(sim_data, sim_hrf, fista_results):
    y = np.load(sim_data, allow_pickle=True)
    hrf_matrix = np.load(sim_hrf, allow_pickle=True)

    # Run FISTA without pylops
    estimates, lambda_ = fista.fista(hrf_matrix, y, group=0.2, use_pylops=False, max_iter=50)

    # Compare estimates
    assert np.allclose(lambda_, np.repeat(np.array([0.60157822]), y.shape[1], axis=0))
    assert np.allclose(estimates, np.load(fista_results, allow_pickle=True), atol=1e-6)


def test_fista_positives(sim_data, sim_hrf, fista_positives):
    y = np.load(sim_data, allow_pickle=True)
    hrf_matrix = np.load(sim_hrf, allow_pickle=True)

    # Run FISTA without pylops
    estimates, lambda_ = fista.fista(
        hrf_matrix, y, group=0.2, use_pylops=False, max_iter=50, positive_only=True
    )

    # Compare estimates
    assert np.allclose(lambda_, np.repeat(np.array([0.60157822]), y.shape[1], axis=0))
    assert np.allclose(
        estimates, np.maximum(np.load(fista_positives, allow_pickle=True), 0), atol=1e-6
    )


def test_fista_pylops(sim_data, sim_hrf, pylops_results):
    y = np.load(sim_data, allow_pickle=True)
    hrf_matrix = np.load(sim_hrf, allow_pickle=True)

    # Run FISTA with pylops
    estimates, lambda_ = fista.fista(hrf_matrix, y, group=0.2, use_pylops=True, max_iter=50)

    # Compare estimates
    assert np.allclose(lambda_, np.repeat(np.array([0.60157822]), y.shape[1], axis=0))
    assert np.allclose(estimates, np.load(pylops_results, allow_pickle=True), atol=1e-6)
