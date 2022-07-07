import numpy as np

from pySPFM.deconvolution import hrf_generator


def test_HRF_matrix(hrf_file, hrf_linear_file):
    hrf_object = hrf_generator.HRFMatrix(TR=1, TE=[0], n_scans=168, block=False)
    hrf_object.generate_hrf()
    hrf_loaded = np.loadtxt(hrf_file)
    assert np.allclose(hrf_object.hrf, hrf_loaded)
    hrf_linear = hrf_generator.HRFMatrix(TR=1, TE=[0], is_afni=False, n_scans=168, block=False)
    hrf_linear.generate_hrf()
    assert np.allclose(hrf_linear.hrf, np.loadtxt(hrf_linear_file))
    hrf_block = hrf_generator.HRFMatrix(TR=1, TE=[0], n_scans=168, block=True)
    hrf_block.generate_hrf()
    assert np.allclose(hrf_block.hrf, np.matmul(hrf_loaded, np.tril(np.ones(hrf_loaded.shape[0]))))
