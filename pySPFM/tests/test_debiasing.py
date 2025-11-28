import numpy as np
import pytest

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


def test_debiasing_spike_no_group(sim_data, sim_hrf):
    """Test debiasing_spike without grouping."""
    sim_data = np.load(sim_data, allow_pickle=True)[:, 0]
    sim_data = sim_data[:, np.newaxis]

    nt = sim_data.shape[0]

    from pySPFM.deconvolution.hrf_generator import HRFMatrix

    hrf_generator = HRFMatrix(block=False, te=[0.018, 0.036, 0.052])
    sim_hrf = hrf_generator.generate_hrf(1, sim_data.shape[0]).hrf_

    sim_data = np.repeat(sim_data, 3, axis=0)

    estimates = np.zeros((nt, 1))
    estimates_idxs = [35, 50, 65, 80]
    estimates[estimates_idxs] = 1

    beta, fitts = debiasing.debiasing_spike(
        sim_hrf,
        sim_data,
        estimates,
        group=False,  # No grouping
    )

    assert beta.shape == estimates.shape
    assert fitts.shape[0] == sim_data.shape[0]


def test_group_hrf():
    """Test group_hrf function."""
    nt = 50
    hrf = np.eye(nt)
    index_events = np.array([5, 6, 7, 15, 16, 30])
    group_dist = 3

    hrf_grouped, grouped_indices = debiasing.group_hrf(hrf, index_events, group_dist)

    # Should have fewer columns than original events
    assert hrf_grouped.shape[1] <= len(index_events)


def test_group_betas():
    """Test group_betas function."""
    nt = 50
    beta = np.zeros(nt)
    beta[5] = 1.0
    beta[15] = 2.0
    beta[30] = 3.0

    index_events = np.array([5, 6, 7, 15, 16, 30])
    group_dist = 3

    grouped_beta = debiasing.group_betas(beta, index_events, group_dist)

    assert grouped_beta.shape == beta.shape


def test_innovation_to_block(sim_data, sim_hrf):
    """Test innovation_to_block function."""
    sim_data = np.load(sim_data, allow_pickle=True)[:, 0]
    sim_data = sim_data[:, np.newaxis]
    sim_hrf = np.load(sim_hrf, allow_pickle=True)

    nt = sim_data.shape[0]
    sim_hrf = np.dot(sim_hrf, np.tril(np.ones(nt)))

    estimates = np.zeros((nt, 1))
    estimates[10] = 1
    estimates[20] = 1
    estimates[30] = 1

    # innovation_to_block takes hrf, y, estimates_matrix, is_ls
    beta, s = debiasing.innovation_to_block(sim_hrf, sim_data[:, 0], estimates[:, 0], is_ls=True)

    assert beta.shape == estimates[:, 0].shape


def test_innovation_to_block_ridge(sim_data, sim_hrf):
    """Test innovation_to_block function with ridge regression."""
    sim_data = np.load(sim_data, allow_pickle=True)[:, 0]
    sim_data = sim_data[:, np.newaxis]
    sim_hrf = np.load(sim_hrf, allow_pickle=True)

    nt = sim_data.shape[0]
    sim_hrf = np.dot(sim_hrf, np.tril(np.ones(nt)))

    estimates = np.zeros((nt, 1))
    estimates[10] = 1
    estimates[20] = 1
    estimates[30] = 1

    # innovation_to_block with ridge regression (is_ls=False)
    beta, s = debiasing.innovation_to_block(sim_hrf, sim_data[:, 0], estimates[:, 0], is_ls=False)

    assert beta.shape == estimates[:, 0].shape


def test_innovation_to_block_no_nonzero(sim_hrf):
    """Test innovation_to_block with no nonzero estimates."""
    sim_hrf = np.load(sim_hrf, allow_pickle=True)

    nt = sim_hrf.shape[0]
    sim_hrf = np.dot(sim_hrf, np.tril(np.ones(nt)))

    y = np.random.randn(nt)
    estimates = np.zeros(nt)

    beta, s = debiasing.innovation_to_block(sim_hrf, y, estimates, is_ls=True)

    # Should return zeros when no nonzero estimates
    assert np.all(beta == 0)


def test_debiasing_spike_empty_estimates():
    """Test debiasing_spike with all-zero estimates."""
    nt = 50
    hrf = np.eye(nt)
    y = np.random.randn(nt, 1)
    estimates = np.zeros((nt, 1))

    beta, fitts = debiasing.debiasing_spike(hrf, y, estimates)

    # Should return zeros
    assert np.all(beta == 0)
