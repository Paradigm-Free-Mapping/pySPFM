import numpy as np

from pySPFM.deconvolution import debiasing


def test_debiasing_spike(sim_data, sim_hrf):
    sim_data = np.load(sim_data, allow_pickle=True)[:, 0]
    sim_data = sim_data[:, np.newaxis]
    sim_hrf = np.load(sim_hrf, allow_pickle=True)

    nt = sim_data.shape[0]

    from pySPFM.deconvolution.hrf_generator import HRFMatrix

    hrf_generator = HRFMatrix(block=False, te=[0.018, 0.036, 0.052])
    sim_hrf = hrf_generator.generate_hrf(1, sim_data.shape[0]).hrf_

    sim_data = np.repeat(sim_data, 3, axis=0)

    estimates = np.zeros((nt, 1))
    estimates_idxs = [
        35,
        50,
        65,
        80,
        151,
        152,
        153,
        154,
        155,
        156,
        157,
        158,
        159,
        160,
        210,
        211,
        212,
        213,
        214,
    ]
    estimates[estimates_idxs] = 1

    beta, fitts = debiasing.debiasing_spike(
        sim_hrf,
        sim_data,
        estimates,
        group=True,
    )
    # Check that beta is different from zero on the same indices as estimates
    assert np.all(beta[estimates != 0] != 0)

    # Check that the convolution of the hrf with estimates is different from zero on the same
    # indices as fitts
    assert np.all(fitts[np.dot(sim_hrf, estimates) != 0] != 0)


def test_debiasing_block(sim_data, sim_hrf):
    sim_data = np.load(sim_data, allow_pickle=True)[:, 0]
    sim_data = sim_data[:, np.newaxis]
    sim_hrf = np.load(sim_hrf, allow_pickle=True)

    nt = sim_data.shape[0]
    sim_hrf = np.dot(sim_hrf, np.tril(np.ones(nt)))

    estimates = np.zeros((nt, 1))
    estimates[35] = 1
    estimates[50] = 1
    estimates[65] = 1
    estimates[80] = 1
    estimates[151:161] = 1
    estimates[210:215] = 1

    beta = debiasing.debiasing_block(sim_hrf, sim_data, estimates_matrix=estimates)

    # Check that the derivative of beta is equal to estimates
    assert np.allclose(np.diff(beta) != 0, estimates != 0)
