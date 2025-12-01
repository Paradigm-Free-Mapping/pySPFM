import os

import numpy as np
import pytest

from pySPFM._solvers import hrf_generator


def test_HRF_matrix(spm_single_echo, spm_single_echo_block, glover_multi_echo):
    hrf_object = hrf_generator.HRFMatrix(te=[0], block=False)
    hrf = hrf_object.generate_hrf(tr=1, n_scans=168).hrf_
    hrf_loaded = np.load(spm_single_echo)
    assert np.allclose(hrf, hrf_loaded)

    hrf_object = hrf_generator.HRFMatrix(te=[0], block=True)
    hrf = hrf_object.generate_hrf(tr=1, n_scans=168).hrf_
    hrf_loaded = np.load(spm_single_echo_block)
    assert np.allclose(hrf, hrf_loaded)

    te = np.array([18.4, 32.0, 48.4])
    hrf_object = hrf_generator.HRFMatrix(te=te, block=False, model="glover")
    hrf = hrf_object.generate_hrf(tr=1, n_scans=168).hrf_
    hrf_loaded = np.load(glover_multi_echo)
    assert np.allclose(hrf, hrf_loaded)


def test_no_te():
    hrf_object = hrf_generator.HRFMatrix(block=False)
    hrf = hrf_object.generate_hrf(tr=1, n_scans=168).hrf_
    assert hrf.shape == (168, 168)


def test_custom_hrf_from_file(testpath):
    """Test loading custom HRF from a .txt file."""
    # Create a simple custom HRF file with n_scans length
    n_scans = 20
    custom_hrf_path = os.path.join(testpath, "custom_hrf.txt")
    custom_hrf = np.zeros(n_scans)
    custom_hrf[:7] = [0.0, 0.2, 0.5, 1.0, 0.8, 0.4, 0.1]  # HRF-like shape
    np.savetxt(custom_hrf_path, custom_hrf)

    hrf_object = hrf_generator.HRFMatrix(te=[0], block=False, model=custom_hrf_path)
    hrf = hrf_object.generate_hrf(tr=1, n_scans=n_scans).hrf_

    # Shape should be (n_scans, n_scans)
    assert hrf.shape == (n_scans, n_scans)


def test_custom_hrf_from_1d_file(testpath):
    """Test loading custom HRF from a .1D file."""
    n_scans = 20
    custom_hrf_path = os.path.join(testpath, "custom_hrf.1D")
    custom_hrf = np.zeros(n_scans)
    custom_hrf[:7] = [0.0, 0.2, 0.5, 1.0, 0.8, 0.4, 0.1]
    np.savetxt(custom_hrf_path, custom_hrf)

    hrf_object = hrf_generator.HRFMatrix(te=[0], block=False, model=custom_hrf_path)
    hrf = hrf_object.generate_hrf(tr=1, n_scans=n_scans).hrf_

    assert hrf.shape == (n_scans, n_scans)


def test_custom_hrf_too_long(testpath):
    """Test that custom HRF longer than n_scans raises ValueError."""
    custom_hrf_path = os.path.join(testpath, "long_hrf.txt")
    # Create HRF longer than n_scans
    custom_hrf = np.ones(50)
    np.savetxt(custom_hrf_path, custom_hrf)

    hrf_object = hrf_generator.HRFMatrix(te=[0], block=False, model=custom_hrf_path)

    with pytest.raises(ValueError, match="HRF is longer than the number of scans"):
        hrf_object.generate_hrf(tr=1, n_scans=20)


def test_invalid_hrf_model():
    """Test that invalid HRF model raises ValueError."""
    hrf_object = hrf_generator.HRFMatrix(te=[0], block=False, model="invalid_model")

    with pytest.raises(ValueError, match="Model must be either"):
        hrf_object.generate_hrf(tr=1, n_scans=20)


def test_multi_echo_block():
    """Test multi-echo with block model."""
    te = np.array([18.4, 32.0, 48.4])
    hrf_object = hrf_generator.HRFMatrix(te=te, block=True, model="spm")
    hrf = hrf_object.generate_hrf(tr=1, n_scans=50).hrf_

    # Multi-echo block should have shape (n_te * n_scans, n_scans)
    assert hrf.shape == (3 * 50, 50)


def test_te_in_seconds():
    """Test that TE values < 1 are assumed to be in seconds."""
    te = [0.018, 0.032, 0.048]  # TE in seconds
    hrf_object = hrf_generator.HRFMatrix(te=te, block=False, model="spm")

    # Should not be converted since they are < 1
    assert hrf_object.te == te

    hrf = hrf_object.generate_hrf(tr=1, n_scans=50).hrf_
    assert hrf.shape == (3 * 50, 50)
