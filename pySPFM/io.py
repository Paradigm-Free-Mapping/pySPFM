"""I/O."""
import json
import os.path as op
from subprocess import run

import nibabel as nib
from nilearn import masking
from nilearn.input_data import NiftiLabelsMasker

from pySPFM.utils import get_keyword_description


def read_data(data_fn, mask_fn, is_atlas=False):
    """Read data from filename and apply mask.

    Parameters
    ----------
    data_fn : str or path
        Path to data to be read.
    mask_fn : str or path
        Path to mask to be applied.

    Returns
    -------
    data_restruct : (T x S) ndarray
        [description]
    data_header : nib.header
        Header of the input data.
    dims : list
        List with dimensions of data.
    mask_idxs : (S x) ndarray
        Indexes to transform data back to 4D.
    """
    data_img = nib.load(data_fn)
    data_header = data_img.header

    if is_atlas:
        mask = NiftiLabelsMasker(labels_img=mask_fn, standardize=False, strategy="mean")
        data = mask.fit_transform(data_img)
    else:
        mask = nib.load(mask_fn)
        data = masking.apply_mask(data_img, mask)

    return data, data_header, mask


def reshape_data(signal2d, mask):
    """Reshape data from 2D back to 4D.

    Parameters
    ----------
    signal2d : (T x S) ndarray
        Data in 2D.
    mask : Nifti1Image
        Mask.

    Returns
    -------
    signal4d : (S x S x S x T) ndarray
        Data in 4D.
    """
    signal4d = masking.unmask(signal2d, mask)
    return signal4d


def update_header(filename, command):
    """Update history of data to be read with 3dInfo.

    Parameters
    ----------
    filename : str or path
        Path to the file that is getting the header updated.
    command : str
        pySPFM command to add to the header.
    """
    run(f"3dcopy {filename} {filename} -overwrite", shell=True)
    run(f'3dNotes -h "{command}" {filename}', shell=True)


def write_data(data, filename, mask, header, command, is_atlas=False, is_bids=False):
    """Write data into NIFTI file.

    Parameters
    ----------
    data : (T x S)
        Data in 2D.
    filename : str or path
        Name of the output file.
    mask : Nifti1Image
        Mask.
    header : nib.header
        Header of the input data.
    command : str
        pySPFM command to add to the header.
    """
    if is_atlas:
        out_img = mask.inverse_transform(data)
    else:
        reshaped = reshape_data(data, mask)
        out_img = nib.Nifti1Image(reshaped.get_fdata(), None, header=header)
    out_img.to_filename(filename)

    # Update header with AFNI if BIDS is not required
    if not is_bids:
        update_header(filename, command)


def write_json(fname, keywords, out_dir):

    # Create dictionary with all the information
    out_dict = {}

    # Iterate over all the keywords and add their description and method to the dictionary
    for keyword in keywords:
        out_dict[keyword] = {}
        out_dict[keyword]["description"] = get_keyword_description(keyword)
        out_dict[keyword]["method"] = "pySPFM"

    # Create output filename
    outname = f"{fname}_desc-pySPFM.json"

    # Write json file
    with open(op.join(out_dir, outname), "w") as f:
        f.write(json.dumps(out_dict, indent=4))
