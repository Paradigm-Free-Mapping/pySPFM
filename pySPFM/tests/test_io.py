import json
import os.path as op

import nibabel as nib
import numpy as np
import pytest
from nilearn.maskers import NiftiMasker

from pySPFM import io


def test_read_data(nih_five_echo_1, mask_five_echo):
    data, masker = io.read_data(nih_five_echo_1, mask_five_echo)
    assert data.shape == (75, 4)
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


def test_write_data_with_bids(nih_five_echo_1, mask_five_echo, testpath):
    """Test write_data with BIDS format (skips AFNI header update)."""
    orig_img = nib.load(nih_five_echo_1)
    dummy_data = np.ones((75, 4))
    out_name = op.join(testpath, "test_write_data_bids.nii.gz")

    masker = NiftiMasker(mask_img=mask_five_echo, standardize=False)
    masker.fit()

    io.write_data(dummy_data, out_name, masker, orig_img, "test_command", use_bids=True)

    assert op.exists(out_name)


def test_write_data_with_string_path(nih_five_echo_1, mask_five_echo, testpath):
    """Test write_data when orig_img is a string path."""
    dummy_data = np.ones((75, 4))
    out_name = op.join(testpath, "test_write_data_str.nii.gz")

    masker = NiftiMasker(mask_img=mask_five_echo, standardize=False)
    masker.fit()

    # Pass string path instead of nibabel image
    io.write_data(dummy_data, out_name, masker, nih_five_echo_1, "test_command", use_bids=True)

    assert op.exists(out_name)


def test_write_data_with_txt_string_path(testpath):
    """Test write_data when orig_img is a txt file path."""
    # Create a txt data file
    txt_path = op.join(testpath, "orig_data.txt")
    data = np.random.randn(50, 10)
    np.savetxt(txt_path, data)

    # Load the txt file to create masker
    data_img, mask_img = io.txt_to_nifti(txt_path)

    masker = NiftiMasker(mask_img=mask_img, standardize=False)
    masker.fit()

    dummy_data = np.ones((50, 10))
    out_name = op.join(testpath, "test_write_data_txt_orig.nii.gz")

    # Pass txt file path as orig_img
    io.write_data(dummy_data, out_name, masker, txt_path, "test_command", use_bids=True)

    assert op.exists(out_name)


def test_txt_to_nifti_txt(testpath):
    """Test txt_to_nifti with .txt file."""
    # Create a simple txt file with space delimiter
    txt_path = op.join(testpath, "test_data.txt")
    data = np.random.randn(50, 10)  # 50 timepoints, 10 voxels
    np.savetxt(txt_path, data, delimiter=" ")

    data_img, mask_img = io.txt_to_nifti(txt_path)

    # Data should be 4D
    assert data_img.ndim == 4
    # Shape should be (1, 1, n_voxels, n_timepoints)
    assert data_img.shape[-1] == 50  # timepoints last dimension


def test_txt_to_nifti_csv(testpath):
    """Test txt_to_nifti with .csv file."""
    csv_path = op.join(testpath, "test_data.csv")
    data = np.random.randn(50, 10)
    np.savetxt(csv_path, data, delimiter=",")

    data_img, mask_img = io.txt_to_nifti(csv_path)
    assert data_img.ndim == 4


def test_txt_to_nifti_tsv(testpath):
    """Test txt_to_nifti with .tsv file."""
    tsv_path = op.join(testpath, "test_data.tsv")
    data = np.random.randn(50, 10)
    np.savetxt(tsv_path, data, delimiter="\t")

    data_img, mask_img = io.txt_to_nifti(tsv_path)
    assert data_img.ndim == 4


def test_txt_to_nifti_unknown_extension(testpath):
    """Test txt_to_nifti with unknown file extension (uses None delimiter)."""
    # Create a file with unknown extension but space-separated data
    unknown_path = op.join(testpath, "test_data.dat")
    data = np.random.randn(50, 10)
    np.savetxt(unknown_path, data)  # Default is space separator

    data_img, mask_img = io.txt_to_nifti(unknown_path)
    assert data_img.ndim == 4


def test_read_data_with_txt(testpath):
    """Test read_data with txt file input."""
    # Create txt file
    txt_path = op.join(testpath, "test_read.txt")
    data = np.random.randn(50, 10)
    np.savetxt(txt_path, data, delimiter=" ")

    # For txt files, mask_fn is not used (created internally)
    data_masked, masker = io.read_data(txt_path, None)

    assert data_masked.shape[0] == 50  # n_timepoints


def test_read_data_with_atlas(testpath):
    """Test read_data with atlas mask (max value > 1)."""
    # Create a simple 4D data file
    data_4d = np.random.randn(10, 10, 10, 50).astype(np.float32)
    data_img = nib.Nifti1Image(data_4d, np.eye(4))
    data_path = op.join(testpath, "test_atlas_data.nii.gz")
    nib.save(data_img, data_path)

    # Create atlas mask with 3 regions
    atlas_data = np.zeros((10, 10, 10), dtype=np.int32)
    atlas_data[2:4, 2:4, 2:4] = 1
    atlas_data[5:7, 5:7, 5:7] = 2
    atlas_data[7:9, 7:9, 7:9] = 3
    atlas_img = nib.Nifti1Image(atlas_data, np.eye(4))
    atlas_path = op.join(testpath, "test_atlas.nii.gz")
    nib.save(atlas_img, atlas_path)

    data_masked, masker = io.read_data(data_path, atlas_path)

    # Should have 3 columns (one per region)
    assert data_masked.shape[1] == 3


def test_read_data_invalid_mask(testpath):
    """Test read_data with invalid mask (all zeros)."""
    # Create data file
    data_4d = np.random.randn(10, 10, 10, 50).astype(np.float32)
    data_img = nib.Nifti1Image(data_4d, np.eye(4))
    data_path = op.join(testpath, "test_invalid_mask_data.nii.gz")
    nib.save(data_img, data_path)

    # Create mask with all zeros
    mask_data = np.zeros((10, 10, 10), dtype=np.int32)
    mask_img = nib.Nifti1Image(mask_data, np.eye(4))
    mask_path = op.join(testpath, "test_invalid_mask.nii.gz")
    nib.save(mask_img, mask_path)

    with pytest.raises(ValueError, match="Mask is not binary or an atlas"):
        io.read_data(data_path, mask_path)


def test_write_json(testpath):
    """Test write_json function."""
    keywords = ["bold", "activityInducing", "innovation", "lambda", "MAD", "beta"]

    io.write_json(keywords, testpath)

    json_path = op.join(testpath, "dataset_description.json")
    assert op.exists(json_path)

    # Read and verify content
    with open(json_path) as f:
        content = json.load(f)

    assert "bold" in content
    assert "activityInducing" in content
    assert content["bold"]["units"] == "percent"


def test_write_json_multi_bold(testpath):
    """Test write_json with multiple bold outputs."""
    keywords = ["bold_echo1", "bold_echo2", "activityInducing"]

    io.write_json(keywords, testpath)

    json_path = op.join(testpath, "dataset_description.json")
    assert op.exists(json_path)

    with open(json_path) as f:
        content = json.load(f)

    # activityInducing should have s-1 units when multiple bold outputs
    assert content["activityInducing"]["units"] == "s-1"
