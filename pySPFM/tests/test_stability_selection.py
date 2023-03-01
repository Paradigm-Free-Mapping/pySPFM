import numpy as np

from pySPFM.deconvolution import stability_selection


def test_get_subsampling_indices():
    n_scans = 10
    n_echos = 3

    # Test same mode
    mode = "same"
    subsample_result = np.array([1, 3, 5, 6, 7, 8])
    np.random.seed(200)
    subsample_idx = stability_selection.get_subsampling_indices(n_scans, n_echos, mode)
    assert np.allclose(subsample_idx, subsample_result)

    # Test different mode
    mode = "different"
    subsample_result = np.array([1, 3, 5, 6, 7, 8, 10, 12, 14, 15, 17, 19, 21, 22, 23, 24, 28, 29])
    np.random.seed(200)
    subsample_idx = stability_selection.get_subsampling_indices(n_scans, n_echos, mode)
    assert np.allclose(
        subsample_idx,
        subsample_result,
    )


def test_calculate_auc():
    n_surrogates = 10
    n_lambdas = 10
    lambdas = np.random.rand(n_lambdas)
    coefs = np.random.rand(n_lambdas, n_surrogates)

    # Add zeros to coefs at random positions
    for i in range(n_surrogates):
        zero_idx = np.random.randint(0, n_lambdas)
        coefs[zero_idx:, i] = 0

    # Test if auc is calculated correctly
    auc = stability_selection.calculate_auc(coefs, lambdas, n_surrogates)

    assert auc <= 1.0
    assert auc >= 0.0


def test_stability_selection(sim_data, sim_hrf):
    y = np.load(sim_data, allow_pickle=True)[:, 0]
    hrf_matrix = np.load(sim_hrf, allow_pickle=True)
    n_lambdas = 20
    n_surrogates = 20

    # Test if stability selection works
    auc = stability_selection.stability_selection(hrf_matrix, y, n_lambdas, n_surrogates)
    assert auc.shape[0] == hrf_matrix.shape[1]
    assert np.count_nonzero(auc) > 0
    assert np.count_nonzero(auc) / auc.shape[0] < 0.5
