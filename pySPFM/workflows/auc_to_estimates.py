import argparse
import datetime
import logging
import os
import sys
from os import path as op

import numpy as np
from dask import compute
from dask import delayed as delayed_dask

from pySPFM import __version__, utils
from pySPFM.deconvolution.debiasing import debiasing_block, debiasing_spike
from pySPFM.deconvolution.hrf_generator import HRFMatrix
from pySPFM.io import read_data, write_data, write_json
from pySPFM.utils import dask_scheduler, get_outname

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
        type=str,
        nargs="+",
        help="The name of the file containing fMRI data.",
        required=True,
    )
    required.add_argument(
        "-a",
        "--auc",
        dest="auc_fn",
        type=str,
        help="The name of the file containing AUC data.",
        required=True,
    )
    required.add_argument(
        "-m",
        "--mask",
        dest="mask_fn",
        type=str,
        help="The name of the file containing the mask for the fMRI data.",
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
    required.add_argument(
        "-thr",
        "--threshold",
        dest="thr",
        type=,
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
        type=str,
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
        "-atlas",
        "--atlas",
        dest="is_atlas",
        action="store_true",
        help="Use provided mask as an atlas (default = False).",
        default=False,
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
    out_dir,
    te=[0],
    hrf_model="spm",
    block_model=False,
    n_jobs=4,
    is_atlas=False,
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
    basename = "pySPFM_"
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
        data_masked, data_header, mask = read_data(data_fn[0], mask_fn, is_atlas=is_atlas)
        n_scans = data_masked.shape[0]
        n_voxels = data_masked.shape[1]
    elif n_te > 1:
        # If the first element of data_fn has spaces, it is a list of paths
        # Convert it into a list
        if " " in data_fn[0]:
            data_fn = data_fn[0].split(" ")

        for te_idx in range(n_te):
            data_temp, data_header, mask = read_data(data_fn[te_idx], mask_fn, is_atlas=is_atlas)
            if te_idx == 0:
                data_masked = data_temp
                n_scans = data_temp.shape[0]
                n_voxels = data_temp.shape[1]
            else:
                data_masked = np.concatenate((data_masked, data_temp), axis=0)

            LGR.info(f"{te_idx + 1}/{n_te} echoes...")

    LGR.info("Data read.")

    LGR.info("Reading AUC data...")
    auc, _, _ = read_data(auc_fn, mask_fn, is_atlas=is_atlas)
    LGR.info("AUC data read.")




def _main():
    """auc_to_estimates entry point"""
    command_str = "auc_to_estimates " + " ".join(sys.argv[1:])
    options = _get_parser().parse_args()
    auc_to_estimates(**vars(options), command_str=command_str)


if __name__ == "__main__":
    _main()
