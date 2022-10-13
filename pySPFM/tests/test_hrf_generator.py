import numpy as np

from pySPFM.deconvolution import hrf_generator


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
