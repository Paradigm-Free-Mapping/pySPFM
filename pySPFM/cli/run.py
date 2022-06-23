# -*- coding: utf-8 -*-
"""Parser for pySPFM."""


import argparse

from pySPFM import __version__


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
        help="The name of the file containing fMRI data. ",
        required=True,
    )
    required.add_argument(
        "-m",
        "--mask",
        dest="mask_fn",
        type=str,
        help="The name of the file containing the mask for the fMRI data. ",
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
        "-block",
        "--block",
        dest="block_model",
        action="store_true",
        help="Estimate innovation signals. Default = False.",
        default=False,
    )
    optional.add_argument(
        "-hrf",
        "--hrf",
        dest="hrf_model",
        type=str,
        help="HRF model to use. Default = 'spm'.",
        default="spm",
        choices=["spm", "glover"],
    )
    optional.add_argument(
        "--hrf_custom",
        dest="hrf_custom",
        type=str,
        help="Custom HRF model to use. Default = 'None'.",
        default="None",
    )
    optional.add_argument(
        "--debias",
        dest="debias",
        action="store_true",
        help="Perform debiasing step. Default = False.",
        default=False,
    )
    optional.add_argument(
        "-g",
        "--group",
        dest="group",
        type=float,
        help="Weight of the grouping in space (we suggest not going "
        "higher than 0.3). Default = 0.",
        default=0,
    )
    optional.add_argument(
        "-crit",
        "--criterion",
        dest="criterion",
        type=str,
        choices=["mad", "mad_update", "ut", "lut", "factor", "pcg", "eigval"],
        help="Criteria with which lambda is selected to estimate activity-inducing "
        "and innovation signals.",
        default="mad_update",
    )
    optional.add_argument(
        "-pcg",
        "--percentage",
        dest="pcg",
        type=float,
        help=(
            "Percentage of maximum lambda to use on temporal regularization with FISTA "
            "(default = None)."
        ),
        default=None,
    )
    optional.add_argument(
        "-factor",
        "--factor",
        dest="factor",
        type=float,
        help="Factor to multiply noise estimate for lambda selection.",
        default=1,
    )
    optional.add_argument(
        "-lambda_echo",
        "--lambda_echo",
        dest="lambda_echo",
        type=int,
        help="Number of the TE data to calculate lambda for (default = last TE).",
        default=-1,
    )
    optional.add_argument(
        "--max_iter_factor",
        dest="max_iter_factor",
        type=float,
        help=(
            "Factor of number of samples to use as the maximum number of iterations for LARS "
            "(default = 1.0)."
        ),
        default=1.0,
    )
    optional.add_argument(
        "--max_iter_fista",
        dest="max_iter_fista",
        type=int,
        help="Maximum number of iterations for FISTA (default = 400).",
        default=400,
    )
    optional.add_argument(
        "--min_iter_fista",
        dest="min_iter_fista",
        type=int,
        help="Minimum number of iterations for FISTA (default = 50).",
        default=50,
    )
    optional.add_argument(
        "--max_iter_spatial",
        dest="max_iter_spatial",
        type=int,
        help="Maximum number of iterations for spatial regularization (default = 100).",
        default=100,
    )
    optional.add_argument(
        "--max_iter",
        dest="max_iter",
        type=int,
        help=(
            "Maximum number of iterations for alternating temporal and spatial regularizations "
            "(default = 10)."
        ),
        default=10,
    )
    optional.add_argument(
        "-jobs",
        "--jobs",
        dest="n_jobs",
        type=int,
        help="Number of cores to take to parallelize debiasing step (default = 4).",
        default=4,
    )
    optional.add_argument(
        "-spatial",
        "--spatial_weight",
        dest="spatial_weight",
        type=float,
        help=(
            "Weight for spatial regularization estimates (estimates of temporal regularization "
            "are equal to 1 minus this value). A value of 0 means only temporal regularization "
            "is applied. Default=0"
        ),
        default=0,
    )
    optional.add_argument(
        "--spatial_lambda",
        dest="spatial_lambda",
        type=float,
        help="Lambda for spatial regularization. Default=1",
        default=1,
    )
    optional.add_argument(
        "--spatial_dim",
        dest="spatial_dim",
        type=int,
        help=(
            "Slice-wise regularization with dim = 2; whole-volume regularization with dim=3. "
            "Default = 3."
        ),
        default=3,
    )
    optional.add_argument(
        "-mu",
        "--mu",
        dest="mu",
        type=float,
        help="Step size for spatial regularization (default = 0.01).",
        default=0.01,
    )
    optional.add_argument(
        "-tol",
        "--tolerance",
        dest="tolerance",
        type=float,
        help="Tolerance for FISTA (default = 1e-6).",
        default=1e-6,
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


if __name__ == "__main__":
    raise RuntimeError(
        "pySPFM/cli/run.py should not be run directly;\n"
        "Please `pip install` pySPFM and use the "
        "`pySPFM` command"
    )
