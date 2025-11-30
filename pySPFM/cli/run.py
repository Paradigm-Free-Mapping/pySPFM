"""Main CLI entry point for pySPFM.

This module provides a unified command-line interface for all pySPFM
deconvolution methods, following scikit-learn naming conventions.

Usage
-----
pySPFM <command> [options]

Commands:
    sparse      Run sparse deconvolution (SPFM)
    lowrank     Run low-rank plus sparse decomposition (SPLORA)
    stability   Run stability selection
"""

import argparse
import datetime
import logging
import os
import sys
from os import path as op

import numpy as np

from pySPFM import __version__, utils
from pySPFM._solvers.select_lambda import select_lambda
from pySPFM.decomposition import (
    LowRankPlusSparse,
    SparseDeconvolution,
    StabilitySelection,
)
from pySPFM.io import read_data, write_data

LGR = logging.getLogger("GENERAL")


def _add_common_arguments(parser):
    """Add arguments common to all subcommands."""
    required = parser.add_argument_group("Required Arguments")
    required.add_argument(
        "-i",
        "--input",
        dest="data_fn",
        type=str,
        nargs="+",
        help="Input fMRI data file(s). Multiple files for multi-echo data.",
        required=True,
    )
    required.add_argument(
        "-m",
        "--mask",
        dest="mask_fn",
        type=str,
        help="Mask file for the fMRI data.",
        required=True,
    )
    required.add_argument(
        "-o",
        "--output",
        dest="output",
        type=str,
        help="Output filename prefix (without extension).",
        required=True,
    )
    required.add_argument(
        "--tr",
        dest="tr",
        type=float,
        help="Repetition time (TR) in seconds.",
        required=True,
    )

    optional = parser.add_argument_group("Optional Arguments")
    optional.add_argument(
        "-d",
        "--dir",
        dest="out_dir",
        type=str,
        default=".",
        help="Output directory. Default: current directory.",
    )
    optional.add_argument(
        "--te",
        dest="te",
        nargs="*",
        type=float,
        default=[0],
        help="Echo times in ms for multi-echo data. Default: [0] (single-echo).",
    )
    optional.add_argument(
        "--hrf",
        dest="hrf_model",
        type=str,
        default="spm",
        help="HRF model: 'spm', 'glover', or path to custom HRF file. Default: 'spm'.",
    )
    optional.add_argument(
        "--block",
        dest="block_model",
        action="store_true",
        default=False,
        help="Use block model (estimate innovation signals). Default: spike model.",
    )
    optional.add_argument(
        "-j",
        "--jobs",
        dest="n_jobs",
        type=int,
        default=4,
        help="Number of parallel jobs. Default: 4.",
    )
    optional.add_argument(
        "--bids",
        dest="use_bids",
        action="store_true",
        default=False,
        help="Use BIDS-style output naming convention.",
    )
    optional.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        default=False,
        help="Enable debug logging.",
    )
    optional.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    return parser


def _add_sparse_arguments(parser):
    """Add arguments specific to sparse deconvolution."""
    sparse_opts = parser.add_argument_group("Sparse Deconvolution Options")
    sparse_opts.add_argument(
        "--criterion",
        dest="criterion",
        type=str,
        default="bic",
        choices=["bic", "aic", "mad", "mad_update", "ut", "lut", "factor", "pcg", "eigval"],
        help="Lambda selection criterion. Default: 'bic'.",
    )
    sparse_opts.add_argument(
        "--factor",
        dest="factor",
        type=float,
        default=10.0,
        help="Factor for lambda selection when criterion='factor'. Default: 10.",
    )
    sparse_opts.add_argument(
        "--debias",
        dest="debias",
        action="store_true",
        default=True,
        help="Perform debiasing step. Default: True.",
    )
    sparse_opts.add_argument(
        "--no-debias",
        dest="debias",
        action="store_false",
        help="Skip debiasing step.",
    )
    sparse_opts.add_argument(
        "--group",
        dest="group",
        type=float,
        default=0.0,
        help="Spatial grouping weight (L2,1-norm). Range [0, 1]. Default: 0.",
    )
    sparse_opts.add_argument(
        "--max-iter",
        dest="max_iter",
        type=int,
        default=400,
        help="Maximum iterations for solver. Default: 400.",
    )
    sparse_opts.add_argument(
        "--tol",
        dest="tol",
        type=float,
        default=1e-6,
        help="Convergence tolerance. Default: 1e-6.",
    )
    sparse_opts.add_argument(
        "--positive",
        dest="positive",
        action="store_true",
        default=False,
        help="Enforce non-negative coefficients.",
    )
    sparse_opts.add_argument(
        "--regressors",
        dest="regressors",
        type=str,
        default=None,
        help="Path to file with nuisance regressors (e.g., motion parameters).",
    )
    return parser


def _add_lowrank_arguments(parser):
    """Add arguments specific to low-rank plus sparse decomposition."""
    lr_opts = parser.add_argument_group("Low-Rank + Sparse Options")
    lr_opts.add_argument(
        "--criterion",
        dest="criterion",
        type=str,
        default="mad_update",
        choices=["mad", "mad_update", "ut", "lut", "factor", "pcg", "eigval"],
        help="Lambda selection criterion. Default: 'mad_update'.",
    )
    lr_opts.add_argument(
        "--eigval-threshold",
        dest="eigval_threshold",
        type=float,
        default=0.1,
        help="Eigenvalue threshold for low-rank estimation. Default: 0.1.",
    )
    lr_opts.add_argument(
        "--debias",
        dest="debias",
        action="store_true",
        default=True,
        help="Perform debiasing step. Default: True.",
    )
    lr_opts.add_argument(
        "--no-debias",
        dest="debias",
        action="store_false",
        help="Skip debiasing step.",
    )
    lr_opts.add_argument(
        "--max-iter",
        dest="max_iter",
        type=int,
        default=100,
        help="Maximum iterations for solver. Default: 100.",
    )
    return parser


def _add_stability_arguments(parser):
    """Add arguments specific to stability selection."""
    stab_opts = parser.add_argument_group("Stability Selection Options")
    stab_opts.add_argument(
        "--n-surrogates",
        dest="n_surrogates",
        type=int,
        default=50,
        help="Number of bootstrap surrogates. Default: 50.",
    )
    stab_opts.add_argument(
        "--threshold",
        dest="threshold",
        type=float,
        default=0.6,
        help="Selection threshold. Default: 0.6.",
    )
    stab_opts.add_argument(
        "--debias",
        dest="debias",
        action="store_true",
        default=True,
        help="Perform debiasing step. Default: True.",
    )
    stab_opts.add_argument(
        "--no-debias",
        dest="debias",
        action="store_false",
        help="Skip debiasing step.",
    )
    return parser


def run_sparse(args, command_str=None):
    """Run sparse deconvolution."""
    LGR.info("Running sparse deconvolution (SparseDeconvolution)")

    # Read data
    data, masker = read_data(args.data_fn[0], args.mask_fn)

    # Create and fit estimator
    model = SparseDeconvolution(
        tr=args.tr,
        te=args.te,
        hrf_model=args.hrf_model,
        block_model=args.block_model,
        criterion=args.criterion,
        debias=args.debias,
        group=args.group,
        max_iter=args.max_iter,
        tol=args.tol,
        n_jobs=args.n_jobs,
        positive=args.positive,
    )

    LGR.info("Fitting model...")
    model.fit(data)
    LGR.info("Model fitted successfully.")

    # Save outputs
    _save_outputs(model, args, masker, data, command_str=command_str)


def run_lowrank(args, command_str=None):
    """Run low-rank plus sparse decomposition."""
    LGR.info("Running low-rank + sparse decomposition (LowRankPlusSparse)")

    # Read data
    data, masker = read_data(args.data_fn[0], args.mask_fn)

    # Create and fit estimator
    model = LowRankPlusSparse(
        tr=args.tr,
        te=args.te,
        hrf_model=args.hrf_model,
        block_model=args.block_model,
        criterion=args.criterion,
        eigval_threshold=args.eigval_threshold,
        debias=args.debias,
        max_iter=args.max_iter,
        n_jobs=args.n_jobs,
    )

    LGR.info("Fitting model...")
    model.fit(data)
    LGR.info(f"Model fitted in {model.n_iter_} iterations.")

    # Save outputs
    _save_outputs(model, args, masker, data, save_lowrank=True, command_str=command_str)


def run_stability(args, command_str=None):
    """Run stability selection."""
    LGR.info("Running stability selection (StabilitySelection)")

    # Read data
    data, masker = read_data(args.data_fn[0], args.mask_fn)

    # Create and fit estimator
    model = StabilitySelection(
        tr=args.tr,
        te=args.te,
        hrf_model=args.hrf_model,
        block_model=args.block_model,
        n_surrogates=args.n_surrogates,
        threshold=args.threshold,
        n_jobs=args.n_jobs,
    )

    LGR.info("Fitting model...")
    model.fit(data)
    LGR.info("Model fitted successfully.")

    # Save outputs
    out_dir = op.abspath(args.out_dir)
    if not op.isdir(out_dir):
        os.makedirs(out_dir)

    # Save command string to call.sh
    if command_str is not None:
        with open(op.join(out_dir, "call.sh"), "w") as f:
            f.write(command_str)

    # Save selection frequencies (AUC)
    output_name = f"{args.output}_pySPFM_AUC.nii.gz"
    write_data(
        model.selection_frequency_,
        op.join(out_dir, output_name),
        masker,
        args.data_fn[0],
        command_str,
        args.use_bids,
    )
    LGR.info(f"Saved selection frequencies (AUC) to {output_name}")


def _save_outputs(model, args, masker, data, save_lowrank=False, command_str=None):
    """Save model outputs to disk."""
    out_dir = op.abspath(args.out_dir)
    if not op.isdir(out_dir):
        os.makedirs(out_dir)

    # Save command string to call.sh
    if command_str is not None:
        with open(op.join(out_dir, "call.sh"), "w") as f:
            f.write(command_str)

    # Save coefficients (activity-inducing signals)
    output_name = f"{args.output}_pySPFM_activityInducing.nii.gz"
    write_data(
        model.coef_,
        op.join(out_dir, output_name),
        masker,
        args.data_fn[0],
        command_str,
        args.use_bids,
    )
    LGR.info(f"Saved activity-inducing signal to {output_name}")

    # Save fitted signal (denoised bold)
    fitted = model.get_fitted_signal()
    output_name = f"{args.output}_pySPFM_denoised_bold.nii.gz"
    write_data(
        fitted,
        op.join(out_dir, output_name),
        masker,
        args.data_fn[0],
        command_str,
        args.use_bids,
    )
    LGR.info(f"Saved denoised BOLD signal to {output_name}")

    # Save lambda map
    if hasattr(model, "lambda_"):
        output_name = f"{args.output}_pySPFM_lambda.nii.gz"
        write_data(
            np.expand_dims(model.lambda_, axis=0),
            op.join(out_dir, output_name),
            masker,
            args.data_fn[0],
            command_str,
            args.use_bids,
        )
        LGR.info(f"Saved lambda map to {output_name}")

    # Save noise estimate (MAD)
    if hasattr(model, "hrf_matrix_") and model.hrf_matrix_ is not None:
        _, _, noise_estimate = select_lambda(hrf=model.hrf_matrix_, y=data)
        output_name = f"{args.output}_pySPFM_MAD.nii.gz"
        write_data(
            np.expand_dims(noise_estimate, axis=0),
            op.join(out_dir, output_name),
            masker,
            args.data_fn[0],
            command_str,
            args.use_bids,
        )
        LGR.info(f"Saved noise estimate (MAD) to {output_name}")

    # Save low-rank component if applicable
    if save_lowrank and hasattr(model, "low_rank_"):
        output_name = f"{args.output}_pySPFM_lowrank.nii.gz"
        write_data(
            model.low_rank_,
            op.join(out_dir, output_name),
            masker,
            args.data_fn[0],
            command_str,
            args.use_bids,
        )
        LGR.info(f"Saved low-rank component to {output_name}")


def main():
    """Entry point for the pySPFM CLI."""
    # Build command string for reproducibility
    command_str = "pySPFM " + " ".join(sys.argv[1:])

    parser = argparse.ArgumentParser(
        prog="pySPFM",
        description=(
            "pySPFM: Sparse hemodynamic deconvolution of fMRI data.\n\n"
            "Available commands:\n"
            "  sparse      Sparse Paradigm Free Mapping (SPFM)\n"
            "  lowrank     Low-Rank plus Sparse decomposition (SPLORA)\n"
            "  stability   Stability selection for robust deconvolution"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Deconvolution method")

    # Sparse deconvolution subcommand
    sparse_parser = subparsers.add_parser(
        "sparse",
        help="Sparse Paradigm Free Mapping (SPFM) using LARS or FISTA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_arguments(sparse_parser)
    _add_sparse_arguments(sparse_parser)
    sparse_parser.set_defaults(func=run_sparse)

    # Low-rank + sparse subcommand
    lowrank_parser = subparsers.add_parser(
        "lowrank",
        help="Low-Rank plus Sparse decomposition (SPLORA).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_arguments(lowrank_parser)
    _add_lowrank_arguments(lowrank_parser)
    lowrank_parser.set_defaults(func=run_lowrank)

    # Stability selection subcommand
    stability_parser = subparsers.add_parser(
        "stability",
        help="Stability selection for robust sparse deconvolution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_arguments(stability_parser)
    _add_stability_arguments(stability_parser)
    stability_parser.set_defaults(func=run_stability)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Setup logging
    out_dir = op.abspath(args.out_dir)
    if not op.isdir(out_dir):
        os.makedirs(out_dir)

    start_time = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
    logname = op.join(out_dir, f"pySPFM_{start_time}.tsv")
    refname = op.join(out_dir, "_references.txt")
    utils.setup_loggers(logname, refname, quiet=False, debug=args.debug)

    LGR.info(f"pySPFM version {__version__}")
    LGR.info(f"Command: {args.command}")

    # Run the selected command with command_str for reproducibility
    args.func(args, command_str=command_str)

    LGR.info("pySPFM completed successfully.")
    utils.teardown_loggers()


if __name__ == "__main__":
    main()
