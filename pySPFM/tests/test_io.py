import os.path as op

import nibabel as nib
import numpy as np
from nilearn.maskers import NiftiMasker

from pySPFM import io


def test_read_data(nih_five_echo_1, mask_five_echo):
    data, header, masker = io.read_data(nih_five_echo_1, mask_five_echo)
    assert data.shape == (75, 4)
    assert header.get_data_shape() == (41, 52, 28, 75)
    assert masker.mask_img.shape == (41, 52, 28)


def test_write_data(nih_five_echo_1, mask_five_echo, testpath):
    orig_img = nib.load(nih_five_echo_1)
    dummy_data = np.ones((75, 4))
    out_name = op.join(testpath, "test_write_data.nii.gz")

    masker = NiftiMasker(mask_img=mask_five_echo, standardize=False)
    masker.fit()

    io.write_data(dummy_data, out_name, masker, orig_img, "test_write_data")

    # Check that the file was created
    assert op.exists(out_name)
