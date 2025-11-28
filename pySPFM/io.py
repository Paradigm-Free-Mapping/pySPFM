"""Input and output functions for pySPFM."""

import json
import logging
import os.path as op
from pathlib import Path
from subprocess import run

import nibabel as nib
import numpy as np
from nilearn.image import new_img_like
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker

from pySPFM.utils import get_keyword_description

LGR = logging.getLogger("GENERAL")


def txt_to_nifti(data_fn):
    """Load txt file and create a nifti image and a mask from that.

    Parameters
    ----------
    data_fn : str or path
        Path to txt data to be read. Assumes columns are "voxels" and rows are "timepoints"

    Returns
    -------
    data_img : nibabel.nifti1.Nifti1Image
        Nifti image equivalent to input txt.
    mask_img : nibabel.nifti1.Nifti1Image
        Nifti image with data mask.

    """
    # Check extention 0 in case of compressed files (handled np by genfromtxt) to set delimiter
    if Path(data_fn).suffixes[0] in [".txt", ".1d", ".par"]:
        delimiter = " "
    elif Path(data_fn).suffixes[0] == ".csv":
        delimiter = ","
    elif Path(data_fn).suffixes[0] == ".tsv":
        delimiter = "\t"
    else:
        delimiter = None

    # Assume data always has time as first dimension, so transpose for nifti 4D
    data = np.genfromtxt(data_fn, delimiter=delimiter).transpose()

    # Make data a 4D nifti img file (time is last dimension)
    while data.ndim < 4:
        data = data[np.newaxis, :]
    LGR.info(f'Loading txt file with {data.shape[2]} "voxels" and {data.shape[3]} timepoints.')

    data_img = nib.Nifti1Image(data, np.eye(4))

    # Create mask image
    mask_img = nib.Nifti1Image(np.ones((1, 1, data.size[2])), np.eye(4))

    return data_img, mask_img


def read_data(data_fn, mask_fn):
    """Read data from filename and apply mask.

    Parameters
    ----------
    data_fn : str or path
        Path to data to be read.
    mask_fn : str or path
        Path to mask to be applied.

    Returns
    -------
    data : (T x S) ndarray
        Data in 2D.
    masker : nilearn.maskers.NiftiMasker
        Masker.
    """
    # Read data
    # Checking suffix 0 in case of compressed types that are handled np by genfromtxt
    if Path(data_fn).suffixes[0] in [".txt", ".1d", ".par", ".csv", ".tsv"]:
        data_img, mask_img = txt_to_nifti(data_fn)
        mask_max = 1
    else:
        data_img = nib.load(data_fn)
        # Load mask and calculate maximum value
        mask_img = nib.load(mask_fn)
        mask_max = mask_img.get_fdata().max()

    # Check if mask is binary
    if mask_max > 1:
        LGR.info("Masking data with atlas.")
        masker = NiftiLabelsMasker(labels_img=mask_img, standardize=False, strategy="mean")
    elif mask_max == 1:
        LGR.info("Masking data with binary mask.")
        masker = NiftiMasker(mask_img=mask_img, standardize=False)
    else:
        raise ValueError("Mask is not binary or an atlas.")

    data = masker.fit_transform(data_img)

    return data, masker


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


def write_data(data, filename, masker, orig_img, command, use_bids=False):
    """Write data into NIFTI file.

    Parameters
    ----------
    data : (T x S)
        Data in 2D.
    filename : str or path
        Name of the output file.
    masker : nilearn.maskers.NiftiMasker
        Masker.
    orig_img : nibabel.nifti1.Nifti1Image or str or path
        Original data.
    command : str
        pySPFM command to add to the header.
    use_bids : bool, optional
        Whether to use BIDS format, by default False
    """
    # Only copy header if it's going to get updated by AFNI
    copy_header = False
    if not use_bids:
        copy_header = True

    # If orig_img is a string, load it
    if isinstance(orig_img, str):
        # Checking suffix 0 in case of compressed types that are handled np by genfromtxt
        if Path(orig_img).suffixes[0] in [".txt", ".1d", ".par", ".csv", ".tsv"]:
            orig_img, _ = txt_to_nifti(orig_img)
        else:
            orig_img = nib.load(orig_img)

    # Transform data back to 4D, generate new image and save it
    out_img = masker.inverse_transform(data)
    new_img = new_img_like(
        orig_img, out_img.get_fdata(), affine=orig_img.affine, copy_header=copy_header
    )
    new_img.to_filename(filename)

    # Update header with AFNI if BIDS is not required
    if not use_bids:
        update_header(filename, command)


def write_json(keywords, out_dir):
    """Write dataset description into JSON file.

    Parameters
    ----------
    keywords : list
        List of keywords to be added to the JSON file.
    out_dir : str or path
        Path to the output directory.
    """
    # Create dictionary with all the information
    out_dict = {}

    # Iterate over all the keywords and add their description and method to the dictionary
    for keyword in keywords:
        out_dict[keyword] = {}
        out_dict[keyword]["description"] = get_keyword_description(keyword)
        out_dict[keyword]["method"] = "pySPFM"

        # Add units
        if "bold" in keyword:
            out_dict[keyword]["units"] = "percent"
        elif "activityInducing" in keyword:
            # Count how many keywords have "bold" in their name
            bold_count = 0
            for k in keywords:
                if "bold" in k:
                    bold_count += 1
            if bold_count > 1:
                out_dict[keyword]["units"] = "s-1"

    # Create output filename
    outname = "dataset_description.json"

    # Write json file
    with open(op.join(out_dir, outname), "w") as f:
        f.write(json.dumps(out_dict, indent=4))
