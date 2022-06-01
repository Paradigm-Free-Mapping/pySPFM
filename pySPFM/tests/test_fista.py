import numpy as np

from pySPFM.deconvolution import fista


def test_fista_pylops(sim_data, sim_hrf, pylops_results):

    y = np.load(sim_data, allow_pickle=True)
    hrf_matrix = np.load(sim_hrf, allow_pickle=True)

    # Run FISTA
    estimates, lambda_ = fista.fista(hrf_matrix, y, group=0.2)

    # Compare estimates
    assert lambda_ == 0.73388091
    assert np.allclose(estimates, np.load(pylops_results, allow_pickle=True), atol=1e-6)
