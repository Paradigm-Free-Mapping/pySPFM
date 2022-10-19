import argparse
import datetime
import logging
import os
import sys
from os import path as op

import nibabel as nib
import numpy as np

from pySPFM import __version__, utils
from pySPFM.deconvolution import debiasing, hrf_generator
from pySPFM.io import read_data, write_data, write_json
from pySPFM.utils import get_outname
from pySPFM.workflows.parser_utils import check_hrf_value, is_valid_file

LGR = logging.getLogger("GENERAL")
RefLGR = logging.getLogger("REFERENCES")


def _get_parser():
    """
    Parse command line inputs for this function.

    Returns
    -------
    parser.parse_args() : argparse dict

    Notes
    -----
    # Argument parser follow template provided by RalphyZ.
    # https://stackoverflow.com/a/43456577
    """
    parser = argparse.ArgumentParser()
    optional = parser._action_groups.pop()
    required = parser.add_argument_group("Required Argument:")
    required.add_argument(
        "-i",
        "--input",
        dest="data_fn",
        type=lambda x: is_valid_file(parser, x),
        nargs="+",
        help="The name of the file containing fMRI data.",
        required=True,
    )
    required.add_argument(
        "-a",
        "--auc",
        dest="auc_fn",
        type=lambda x: is_valid_file(parser, x),
        help="The name of the file containing AUC data.",
        required=True,
    )
    required.add_argument(
        "-m",
        "--mask",
        dest="mask_fn",
        nargs="+",
        type=lambda x: is_valid_file(parser, x),
        help=(
            "The name of the files containing the mask for the fMRI data and the AUC "
            "thresholding mask."
        ),
        required=True,
    )
    required.add_argument(
        "-o",
        "--output",
        dest="output_filename",
        type=str,
        help="The name of the output file with no extension.",
        required=True,
    )
    required.add_argument(
        "-tr",
        dest="tr",
        type=float,
        help="TR of the fMRI data acquisition.",
        required=True,
    )
    optional.add_argument(
        "-thr",
        "--threshold",
        dest="thr",
        help=(
            "Percentile to threshold the AUC data with. The percentile is applied to the second "
            "mask provided with the '-m' flag if the second mask is a binary mask. If the second "
            "mask is not binary, the values on the second mask are used as the threshold. "
            "Default is 95."
        ),
        type=int,
        default=95,
    )
    optional.add_argument(
        "--strategy",
        dest="thr_strategy",
        help=(
            "Strategy to threshold the AUC data with. If the second mask is a binary mask, "
            "the can be applied with a static threshold ('static') or a time-dependet threshold "
            "('time'). Default is 'static'."
        ),
        type=str,
        default="static",
        choices=["static", "time"],
    )
    optional.add_argument(
        "-d",
        "--dir",
        dest="out_dir",
        type=str,
        help="Output directory. Default is current.",
        default=".",
    )
    optional.add_argument(
        "-te",
        dest="te",
        nargs="*",
        type=float,
        help="List with TE of the fMRI data acquisition.",
        default=[0],
    )
    optional.add_argument(
        "-hrf",
        "--hrf",
        dest="hrf_model",
        type=check_hrf_value,
        help=(
            "HRF model to use. Default is 'spm'. Options are 'spm', 'glover', or a custom HRF "
            "file with the '.1D' or '.txt' extension."
        ),
        default="spm",
    )
    optional.add_argument(
        "-block",
        "--block",
        dest="block_model",
        action="store_true",
        help="Estimate innovation signals. Default = False.",
        default=False,
    )
    optional.add_argument(
        "-jobs",
        "--jobs",
        dest="n_jobs",
        type=int,
        help="Number of jobs to parallelize for loops (default = 4).",
        default=4,
    )
    optional.add_argument(
        "-bids",
        "--bids",
        dest="use_bids",
        action="store_true",
        help=(
            "Use BIDS-style suffix on the given `output` (default = False). pySPFM assumes that "
            "`output` follows the BIDS convention. Not using this option will default to using "
            "AFNI to update the header of the output."
        ),
        default=False,
    )
    optional.add_argument(
        "-debug",
        "--debug",
        dest="debug",
        action="store_true",
        help="Logs in the terminal will have increased "
        "verbosity, and will also be written into "
        "a .tsv file in the output directory.",
        default=False,
    )
    optional.add_argument(
        "-quiet",
        "--quiet",
        dest="quiet",
        help=argparse.SUPPRESS,
        action="store_true",
        default=False,
    )
    optional.add_argument("-v", "--version", action="version", version=("%(prog)s " + __version__))

    parser._action_groups.append(optional)

    return parser


def auc_to_estimates(
    data_fn,
    auc_fn,
    mask_fn,
    output_filename,
    tr,
    thr=95,
    thr_strategy="static",
    out_dir=".",
    te=[0],
    hrf_model="spm",
    block_model=False,
    n_jobs=4,
    use_bids=False,
    debug=False,
    quiet=False,
    command_str=None,
):
    # Generate output directory if it doesn't exist
    out_dir = op.abspath(out_dir)
    if not op.isdir(out_dir):
        os.mkdir(out_dir)

    # Save command into sh file, if the command-line interface was used
    if command_str is not None:
        command_file = open(os.path.join(out_dir, "call.sh"), "w")
        command_file.write(command_str)
        command_file.close()

    LGR = logging.getLogger("GENERAL")
    # RefLGR = logging.getLogger("REFERENCES")
    # create logfile name
    basename = "auc_to_estimates_"
    extension = "tsv"
    start_time = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
    logname = op.join(out_dir, (basename + start_time + "." + extension))
    refname = op.join(out_dir, "_references.txt")
    utils.setup_loggers(logname, refname, quiet=quiet, debug=debug)

    LGR.info("Using output directory: {}".format(out_dir))

    n_te = len(te)

    if all(i >= 1 for i in te):
        te = [x / 1000 for x in te]

    LGR.info("Reading data...")
    if n_te == 1:
        data_masked, masker = read_data(data_fn[0], mask_fn[0])
        n_scans = data_masked.shape[0]
    elif n_te > 1:
        # If the first element of data_fn has spaces, it is a list of paths
        # Convert it into a list
        if " " in data_fn[0]:
            data_fn = data_fn[0].split(" ")

        for te_idx in range(n_te):
            data_temp, masker = read_data(data_fn[te_idx], mask_fn[0])
            if te_idx == 0:
                data_masked = data_temp
                n_scans = data_temp.shape[0]
            else:
                data_masked = np.concatenate((data_masked, data_temp), axis=0)

            LGR.info(f"{te_idx + 1}/{n_te} echoes...")

    LGR.info("Data read.")

    LGR.info("Reading AUC data...")
    auc, _, _ = read_data(auc_fn, mask_fn[0])
    LGR.info("AUC data read.")

    # Threshold the AUC if thr is not 0 and mask_fn has two elements
    if thr != 0 and len(mask_fn) == 2:
        # Read the second mask
        auc_mask = nib.load(mask_fn[1])

        # If the mask is 3D, then it is a binary mask or a static threshold
        if len(auc_mask.shape) == 3:
            # If the mask is binary, then read the AUC values inside of the mask
            if np.max(auc_mask.get_fdata()) == 1:

                auc_thr_values = read_data(mask_fn[1], mask_fn[0])[0]

                if thr_strategy == "static":
                    LGR.info(
                        f"Thresholding AUC values with a {thr}th percentile static threshold..."
                    )
                    # Threshold the whole-brain AUC based on the thr percentile of the AUC values
                    # in the mask
                    auc_thr = auc - np.percentile(auc_thr_values, thr)
                    auc_thr[auc_thr < 0] = 0
                else:
                    LGR.info(
                        f"Thresholding AUC values with a {thr}th percentile time-dependet "
                        "threshold..."
                    )

                    # Calculate and apply percentile at each TR
                    auc_thr = np.zeros(auc.shape)
                    for tr_idx in range(n_scans):
                        auc_thr[tr_idx, :] = auc[tr_idx, :] - np.percentile(
                            auc_thr_values[tr_idx, :], thr
                        )
                        auc_thr[tr_idx, auc_thr[tr_idx, :] < 0] = 0

            # If the mask is a static threshold, then apply it to the AUC values
            else:
                LGR.info("Thresholding AUC values based on the given 3D threshold...")
                auc_mask_data = masker.fit_transform(auc_mask)

                # Threshold the AUC values
                auc_thr = auc - auc_mask_data
                auc_thr[auc_thr < 0] = 0

        # If the mask is 4D, then it is a time-dependent threshold
        elif len(auc_mask.shape) == 4:
            LGR.info("Thresholding AUC values based on the given 4D threshold...")
            # Read the time-dependent threshold
            auc_mask_data = masker.fit_transform(auc_mask)

            # Threshold the AUC
            auc_thr = auc - auc_mask_data
            auc_thr[auc_thr < 0] = 0
        else:
            raise ValueError("The mask used to threshold the AUC must be 3D or 4D.")

    LGR.info("AUC data thresholded.")

    # Generate design matrix with shifted versions of HRF
    LGR.info("Generating design matrix with shifted versions of HRF...")
    hrf_obj = hrf_generator.HRFMatrix(te=te, block=block_model, model=hrf_model)
    hrf = hrf_obj.generate_hrf(tr=tr, n_scans=n_scans).hrf_

    # Solve ordinary least squares problem to calculate estimates
    LGR.info("Calculating estimates...")
    if block_model:
        estimates_spike = debiasing.debiasing_block(hrf, data_masked, auc_thr, n_jobs)
        fitts = np.dot(hrf, estimates_spike)
    else:
        estimates_spike, fitts = debiasing.debiasing_spike(hrf, data_masked, auc_thr, n_jobs)

    LGR.info("Estimates calculated.")

    # Save estimates and thresholded AUC
    LGR.info("Saving results...")
    out_bids_keywords = []

    # Save thresholded AUC
    out_bids_keywords.append("AUC")
    output_name = get_outname(output_filename, "AUC", "nii.gz", use_bids)
    write_data(
        auc,
        os.path.join(out_dir, output_name),
        masker,
        data_fn[0],
        command_str,
        use_bids,
    )

    # Save activity-inducing signal
    if n_te == 1:
        output_name = get_outname(output_filename, "activityInducing", "nii.gz", use_bids)
        out_bids_keywords.append("activityInducing")
    elif n_te > 1:
        output_name = get_outname(output_filename, "activityInducing", "nii.gz", use_bids)
        out_bids_keywords.append("activityInducing")
    write_data(
        estimates_spike,
        os.path.join(out_dir, output_name),
        masker,
        data_fn[0],
        command_str,
        use_bids=use_bids,
    )

    # Save fitts
    if n_te == 1:
        output_name = get_outname(output_filename, "denoised_bold", "nii.gz", use_bids)
        out_bids_keywords.append("denoised_bold")
        write_data(
            fitts,
            os.path.join(out_dir, output_name),
            masker,
            data_fn[0],
            command_str,
            use_bids=use_bids,
        )
    elif n_te > 1:
        for te_idx in range(n_te):
            te_data = fitts[te_idx * n_scans : (te_idx + 1) * n_scans, :]
            output_name = get_outname(
                f"{output_filename}_echo-{te_idx + 1}", "denoised_bold", "nii.gz", use_bids
            )
            out_bids_keywords.append(f"echo-{te_idx + 1}_desc-denoised_bold")
            write_data(
                te_data,
                os.path.join(out_dir, output_name),
                masker,
                data_fn[0],
                command_str,
                use_bids=use_bids,
            )

    # Save BIDS compatible sidecar file
    if use_bids:
        write_json(out_bids_keywords, out_dir)

    LGR.info("Results saved.")

    LGR.info("auc_to_estimates finished.")
    utils.teardown_loggers()


def _main():
    """auc_to_estimates entry point"""
    command_str = "auc_to_estimates " + " ".join(sys.argv[1:])
    options = _get_parser().parse_args()
    auc_to_estimates(**vars(options), command_str=command_str)


if __name__ == "__main__":
    _main()
