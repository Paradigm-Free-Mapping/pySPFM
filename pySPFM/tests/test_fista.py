import numpy as np

from pySPFM.deconvolution import fista


def test_fista_pylops(sim_data, sim_hrf, pylops_results):

    y = np.load(sim_data, allow_pickle=True)
    hrf_matrix = np.load(sim_hrf, allow_pickle=True)

    # Run FISTA
    estimates, lambda_ = fista.fista(hrf_matrix, y, group=0.2, use_pylops=True)

    # Compare estimates
    assert np.allclose(lambda_, np.repeat(np.array([0.6015782243905021]), y.shape[1], axis=0))
    assert np.allclose(estimates, np.load(pylops_results, allow_pickle=True))
