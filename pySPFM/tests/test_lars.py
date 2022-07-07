import numpy as np

from pySPFM.deconvolution import lars


def test_solve_regularization_path(sim_data, sim_hrf, coef_path_results):

    sim_data = np.load(sim_data, allow_pickle=True)[:, 0]
    sim_data = sim_data[:, np.newaxis]

    sim_hrf = np.load(sim_hrf, allow_pickle=True)

    # BIC
    coef_path, lambda_ = lars.solve_regularization_path(
        sim_hrf, sim_data, n_lambdas=50, criterion="bic"
    )

    true_path = np.load(coef_path_results, allow_pickle=True)
    assert np.allclose(lambda_, 0.003012460125648784)
    assert np.allclose(coef_path, true_path[:, 0])

    # AIC
    coef_path, lambda_ = lars.solve_regularization_path(
        sim_hrf, sim_data, n_lambdas=50, criterion="aic"
    )

    assert np.allclose(lambda_, 0.0017909675614952592)
    assert np.allclose(coef_path, true_path[:, 1])
