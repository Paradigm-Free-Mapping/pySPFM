import os.path as op

import nibabel as nib
import numpy as np

from pySPFM import io


def test_read_data(nih_five_echo_1, mask_five_echo):
    data, header, mask = io.read_data(nih_five_echo_1, mask_five_echo)
    assert data.shape == (75, 4)
    assert header.get_data_shape() == (41, 52, 28, 75)
    assert mask.shape == (41, 52, 28)


def test_reshape_data(mask_five_echo):
    data = np.ones((75, 4))
    data_reshaped = io.reshape_data(data, mask_five_echo)
    assert data_reshaped.shape == (41, 52, 28, 75)


def test_write_data(nih_five_echo_1, mask_five_echo, testpath):
    header = nib.load(nih_five_echo_1).header
    dummy_data = np.ones((75, 4))
    out_name = op.join(testpath, "test_write_data.nii.gz")

    io.write_data(dummy_data, out_name, mask_five_echo, header, "test_write_data", is_atlas=False)

    # Check that the file was created
    assert op.exists(out_name)
