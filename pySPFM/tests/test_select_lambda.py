import numpy as np
import pytest

from pySPFM.deconvolution.select_lambda import select_lambda


def test_select_lambda(sim_data, sim_hrf):
    sim_data = np.load(sim_data, allow_pickle=True)[:, 0]
    sim_hrf = np.load(sim_hrf, allow_pickle=True)
    nt = sim_hrf.shape[1]

    # MAD
    lambda_, update_lambda, noise_estimate = select_lambda(sim_hrf, sim_data, criterion="mad")
    assert np.allclose(lambda_, 0.27027350689245333)
    assert update_lambda is False
    assert np.allclose(noise_estimate, lambda_)

    # MAD update
    lambda_, update_lambda, noise_estimate = select_lambda(
        sim_hrf, sim_data, criterion="mad_update"
    )
    assert np.allclose(lambda_, 0.27027350689245333)
    assert update_lambda
    assert np.allclose(noise_estimate, lambda_)

    # Universal threshold
    lambda_, _, _ = select_lambda(sim_hrf, sim_data, criterion="ut")
    assert np.allclose(lambda_, noise_estimate * np.sqrt(2 * np.log10(nt)))

    # Lower universal threshold
    lambda_, _, _ = select_lambda(sim_hrf, sim_data, criterion="lut")
    assert np.allclose(
        lambda_, noise_estimate * np.sqrt(2 * np.log10(nt) - np.log10(1 + 4 * np.log10(nt)))
    )

    # Factor
    lambda_, _, _ = select_lambda(sim_hrf, sim_data, criterion="factor", factor=10)
    assert np.allclose(lambda_, noise_estimate * 10)

    # Percentage of maximum lambda
    max_lambda = np.mean(abs(np.dot(sim_hrf.T, sim_data)), axis=0)
    print(max_lambda)
    lambda_, _, _ = select_lambda(sim_hrf, sim_data, criterion="pcg", pcg=0.1)
    assert np.allclose(lambda_, max_lambda * 0.1)

    # Test that the code raises an error if pcg is None
    with pytest.raises(ValueError):
        select_lambda(sim_hrf, sim_data, criterion="pcg", pcg=None)

    # Eigenvalue
    lambda_, _, _ = select_lambda(sim_hrf, sim_data[:, np.newaxis], criterion="eigval")
    assert np.round(lambda_, 1) >= 4.0
    assert np.round(lambda_, 1) <= 5.0
