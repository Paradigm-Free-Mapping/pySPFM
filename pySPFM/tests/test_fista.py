import numpy as np

from pySPFM.deconvolution import fista


def test_fista(sim_data, sim_hrf, pylops_results, fista_results):

    y = np.load(sim_data, allow_pickle=True)
    hrf_matrix = np.load(sim_hrf, allow_pickle=True)

    # Run FISTA with pylops
    estimates, lambda_ = fista.fista(hrf_matrix, y, group=0.2, use_pylops=True, max_iter=50)

    # Compare estimates
    assert np.allclose(lambda_, np.repeat(np.array([0.60157822]), y.shape[1], axis=0))
    assert np.allclose(estimates, np.load(pylops_results, allow_pickle=True))

    # Run FISTA without pylops
    estimates, lambda_ = fista.fista(hrf_matrix, y, group=0.2, use_pylops=False, max_iter=50)

    # Compare estimates
    assert np.allclose(lambda_, np.repeat(np.array([0.60157822]), y.shape[1], axis=0))
    assert np.allclose(estimates, np.load(fista_results, allow_pickle=True))
