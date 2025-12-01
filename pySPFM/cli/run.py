"""Command-line interface for pySPFM.

This module provides the entry points for the pySPFM command-line interface.
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import numpy as np

from pySPFM import __version__
from pySPFM.cli._preprocessing import remove_regressors
from pySPFM.io import read_data, write_data, write_json
from pySPFM.utils import setup_loggers

LGR = logging.getLogger("GENERAL")
RefLGR = logging.getLogger("REFERENCES")


def _save_call_script(out_dir, argv):
    """Save the command-line call to a shell script.

    Parameters
    ----------
    out_dir : str
        Output directory.
    argv : list
        Command-line arguments.
    """
    call_file = os.path.join(out_dir, "call.sh")
    with open(call_file, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(" ".join(argv) + "\n")
    os.chmod(call_file, 0o755)


def _add_common_args(parser):
    """Add common arguments to subcommand parsers."""
    parser.add_argument(
        "-i",
        "--input",
        dest="data",
        nargs="+",
        required=True,
        help="Input fMRI data file(s). For multi-echo, provide multiple files.",
    )
    parser.add_argument(
        "-m",
        "--mask",
        dest="mask",
        required=True,
        help="Mask file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="out_prefix",
        required=True,
        help="Output file prefix.",
    )
    parser.add_argument(
        "-d",
        "--out-dir",
        dest="out_dir",
        default=".",
        help="Output directory (default: current directory).",
    )
    parser.add_argument(
        "--tr",
        dest="tr",
        type=float,
        required=True,
        help="Repetition time (TR) in seconds.",
    )
    parser.add_argument(
        "-te",
        "--echo-times",
        dest="te",
        nargs="+",
        type=float,
        default=None,
        help="Echo times in ms for multi-echo data.",
    )
    parser.add_argument(
        "-j",
        "--n-jobs",
        dest="n_jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (default: 1).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )
    parser.add_argument(
        "--hrf",
        dest="hrf_model",
        default="spm",
        help="HRF model to use: 'spm', 'glover', or path to custom HRF file.",
    )
    parser.add_argument(
        "--block",
        action="store_true",
        dest="block_model",
        help="Use block model (innovation signal) instead of spike model.",
    )
    parser.add_argument(
        "--bids",
        action="store_true",
        dest="use_bids",
        help="Use BIDS naming convention for outputs.",
    )


def _add_sparse_args(parser):
    """Add arguments specific to sparse deconvolution."""
    parser.add_argument(
        "--criterion",
        dest="criterion",
        default="bic",
        choices=["bic", "aic", "mad", "mad_update", "ut", "lut", "factor", "pcg", "eigval"],
        help="Criterion for lambda selection (default: bic).",
    )
    parser.add_argument(
        "--max-iter",
        dest="max_iter",
        type=int,
        default=400,
        help="Maximum number of iterations (default: 400).",
    )
    parser.add_argument(
        "--min-iter",
        dest="min_iter",
        type=int,
        default=50,
        help="Minimum number of iterations (default: 50).",
    )
    parser.add_argument(
        "--tol",
        dest="tol",
        type=float,
        default=1e-6,
        help="Convergence tolerance (default: 1e-6).",
    )
    parser.add_argument(
        "--debias",
        action="store_true",
        dest="debias",
        help="Perform debiasing step.",
    )
    parser.add_argument(
        "--group",
        dest="group",
        type=float,
        default=0.0,
        help="Group sparsity weight (default: 0.0).",
    )
    parser.add_argument(
        "--pcg",
        dest="pcg",
        type=float,
        default=0.8,
        help="Percentage of maximum lambda (default: 0.8).",
    )
    parser.add_argument(
        "--factor",
        dest="factor",
        type=float,
        default=1.0,
        help="Factor for noise estimate (default: 1.0).",
    )
    parser.add_argument(
        "--regressors",
        dest="regressors",
        default=None,
        help="Path to confound regressors file (.txt, .csv, or .tsv).",
    )


def _get_parser():
    """Create the main argument parser.

    Returns
    -------
    parser : argparse.ArgumentParser
        The argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="pySPFM",
        description="Sparse Paradigm Free Mapping for fMRI deconvolution.",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Sparse deconvolution subcommand
    sparse_parser = subparsers.add_parser(
        "sparse",
        help="Run sparse deconvolution (SPFM).",
    )
    _add_common_args(sparse_parser)
    _add_sparse_args(sparse_parser)

    # Low-rank + sparse subcommand
    lowrank_parser = subparsers.add_parser(
        "lowrank",
        help="Run low-rank plus sparse deconvolution (SPLORA).",
    )
    _add_common_args(lowrank_parser)
    _add_sparse_args(lowrank_parser)
    lowrank_parser.add_argument(
        "--eigval-threshold",
        dest="eigval_threshold",
        type=float,
        default=0.1,
        help="Eigenvalue threshold for low-rank (default: 0.1).",
    )

    # Stability selection subcommand
    stability_parser = subparsers.add_parser(
        "stability",
        help="Run stability selection.",
    )
    _add_common_args(stability_parser)
    stability_parser.add_argument(
        "--n-surrogates",
        dest="n_surrogates",
        type=int,
        default=50,
        help="Number of surrogates for stability selection (default: 50).",
    )
    stability_parser.add_argument(
        "--n-lambdas",
        dest="n_lambdas",
        type=int,
        default=None,
        help="Number of lambda values (default: n_scans).",
    )
    stability_parser.add_argument(
        "--threshold",
        dest="threshold",
        type=float,
        default=0.6,
        help="Selection threshold (default: 0.6).",
    )

    return parser


def _compute_mad(y, hrf_matrix, coef):
    """Compute Median Absolute Deviation of residuals.

    Parameters
    ----------
    y : ndarray of shape (n_timepoints, n_voxels)
        Input fMRI data.
    hrf_matrix : ndarray of shape (n_timepoints, n_scans)
        HRF convolution matrix.
    coef : ndarray of shape (n_scans, n_voxels)
        Estimated coefficients.

    Returns
    -------
    mad : ndarray of shape (n_voxels,)
        MAD for each voxel.
    """
    fitted = np.dot(hrf_matrix, coef)
    residuals = y - fitted
    mad = 1.4826 * np.median(np.abs(residuals - np.median(residuals, axis=0)), axis=0)
    return mad


def _run_sparse(args, command_str, out_dir):
    """Run sparse deconvolution workflow.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    command_str : str
        The command string for logging.
    out_dir : str
        Output directory.
    """
    from pySPFM.decomposition import SparseDeconvolution

    # Handle echo times
    te = args.te if args.te is not None else [0]

    # Read input data (may be multi-echo)
    data_files = args.data
    n_echoes = len(data_files)

    # Read the first file to get dimensions
    data, masker = read_data(data_files[0], args.mask)
    n_scans, n_voxels = data.shape

    if n_echoes > 1:
        # Multi-echo: concatenate along time axis
        all_data = [data]
        for data_fn in data_files[1:]:
            echo_data, _ = read_data(data_fn, args.mask)
            all_data.append(echo_data)
        data = np.vstack(all_data)

    # Load regressors if provided
    if args.regressors is not None:
        LGR.info(f"Loading regressors from {args.regressors}")
        regressors = np.genfromtxt(args.regressors)
        data = remove_regressors(data, regressors, n_scans, n_echoes)

    # Create and fit the estimator
    model = SparseDeconvolution(
        tr=args.tr,
        te=te,
        hrf_model=args.hrf_model,
        block_model=args.block_model,
        criterion=args.criterion,
        debias=args.debias,
        group=args.group,
        pcg=args.pcg,
        factor=args.factor,
        max_iter=args.max_iter,
        min_iter=args.min_iter,
        tol=args.tol,
        n_jobs=args.n_jobs,
    )

    LGR.info("Fitting sparse deconvolution model...")
    model.fit(data)

    # Generate output filenames
    prefix = args.out_prefix
    use_bids = args.use_bids

    # Track output keywords for BIDS json
    output_keywords = []

    if use_bids:
        # BIDS naming convention
        coef_fn = f"{prefix}_desc-activityInducing.nii.gz"
        lambda_fn = f"{prefix}_desc-stat-lambda_statmap.nii.gz"
        output_keywords.append("activityInducing")

        if args.block_model:
            # Save innovation signal for block model
            innov_fn = f"{prefix}_desc-innovation.nii.gz"
            output_keywords.append("innovation")
    else:
        # Standard naming
        coef_fn = f"{prefix}_pySPFM_activityInducing.nii.gz"
        lambda_fn = f"{prefix}_pySPFM_lambda.nii.gz"

    # Save activity-inducing signal
    coef_path = os.path.join(out_dir, coef_fn)
    write_data(model.coef_, coef_path, masker, data_files[0], command_str, use_bids=use_bids)
    LGR.info(f"Saved activity-inducing signal to {coef_fn}")

    # Save innovation signal for block model
    if args.block_model and use_bids:
        innov_path = os.path.join(out_dir, innov_fn)
        # For block model, innovation is the derivative of activity-inducing
        innovation = np.diff(model.coef_, axis=0, prepend=0)
        write_data(innovation, innov_path, masker, data_files[0], command_str, use_bids=use_bids)
        LGR.info(f"Saved innovation signal to {innov_fn}")

    # Save lambda values
    lambda_path = os.path.join(out_dir, lambda_fn)
    write_data(
        model.lambda_.reshape(1, -1), lambda_path, masker, data_files[0], command_str, use_bids
    )
    LGR.info(f"Saved lambda values to {lambda_fn}")

    # Compute and save denoised BOLD
    denoised = np.dot(model.hrf_matrix_, model.coef_)

    if n_echoes > 1 and use_bids:
        # Multi-echo BIDS: save per-echo denoised and MAD
        for echo_idx in range(n_echoes):
            echo_num = echo_idx + 1
            start_idx = echo_idx * n_scans
            end_idx = (echo_idx + 1) * n_scans

            # Denoised BOLD for this echo
            denoised_echo_fn = f"{prefix}_echo-{echo_num}_desc-denoised_bold.nii.gz"
            denoised_echo_path = os.path.join(out_dir, denoised_echo_fn)
            write_data(
                denoised[start_idx:end_idx, :],
                denoised_echo_path,
                masker,
                data_files[echo_idx],
                command_str,
                use_bids=use_bids,
            )
            LGR.info(f"Saved denoised BOLD for echo {echo_num} to {denoised_echo_fn}")
            output_keywords.append(f"echo-{echo_num}_denoised_bold")

            # MAD for this echo
            mad_echo_fn = f"{prefix}_desc-echo-{echo_num}_MAD.nii.gz"
            mad_echo_path = os.path.join(out_dir, mad_echo_fn)
            mad_echo = _compute_mad(data[start_idx:end_idx, :], model.hrf_matrix_, model.coef_)
            write_data(
                mad_echo.reshape(1, -1),
                mad_echo_path,
                masker,
                data_files[echo_idx],
                command_str,
                use_bids=use_bids,
            )
            LGR.info(f"Saved MAD for echo {echo_num} to {mad_echo_fn}")
            output_keywords.append(f"echo-{echo_num}_MAD")
    else:
        # Single-echo or non-BIDS
        if use_bids:
            denoised_fn = f"{prefix}_desc-denoised_bold.nii.gz"
            mad_fn = f"{prefix}_desc-MAD.nii.gz"
        else:
            denoised_fn = f"{prefix}_pySPFM_denoised_bold.nii.gz"
            mad_fn = f"{prefix}_pySPFM_MAD.nii.gz"

        denoised_path = os.path.join(out_dir, denoised_fn)
        write_data(denoised, denoised_path, masker, data_files[0], command_str, use_bids=use_bids)
        LGR.info(f"Saved denoised BOLD to {denoised_fn}")

        # Compute and save MAD
        mad = _compute_mad(data, model.hrf_matrix_, model.coef_)
        mad_path = os.path.join(out_dir, mad_fn)
        write_data(
            mad.reshape(1, -1), mad_path, masker, data_files[0], command_str, use_bids=use_bids
        )
        LGR.info(f"Saved MAD to {mad_fn}")

    # Write BIDS dataset_description.json if using BIDS
    if use_bids:
        write_json(output_keywords, out_dir)
        LGR.info("Saved dataset_description.json")


def _run_lowrank(args, command_str, out_dir):
    """Run low-rank plus sparse deconvolution workflow.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    command_str : str
        The command string for logging.
    out_dir : str
        Output directory.
    """
    from pySPFM.decomposition import LowRankPlusSparse

    # Handle echo times
    te = args.te if args.te is not None else [0]

    # Read data
    data, masker = read_data(args.data[0], args.mask)

    # Create and fit the estimator
    model = LowRankPlusSparse(
        tr=args.tr,
        te=te,
        hrf_model=args.hrf_model,
        block_model=args.block_model,
        criterion=args.criterion,
        eigval_threshold=args.eigval_threshold,
        debias=args.debias,
        group=args.group,
        factor=args.factor,
        max_iter=args.max_iter,
        min_iter=args.min_iter,
        tol=args.tol,
        n_jobs=args.n_jobs,
    )

    LGR.info("Fitting low-rank plus sparse model...")
    model.fit(data)

    # Generate output filenames
    prefix = args.out_prefix
    use_bids = args.use_bids

    if use_bids:
        coef_fn = f"{prefix}_desc-sparse.nii.gz"
        lowrank_fn = f"{prefix}_desc-lowrank.nii.gz"
        lambda_fn = f"{prefix}_desc-stat-lambda_statmap.nii.gz"
    else:
        coef_fn = f"{prefix}_pySPFM_sparse.nii.gz"
        lowrank_fn = f"{prefix}_pySPFM_lowrank.nii.gz"
        lambda_fn = f"{prefix}_pySPFM_lambda.nii.gz"

    # Save sparse component
    coef_path = os.path.join(out_dir, coef_fn)
    write_data(model.coef_, coef_path, masker, args.data[0], command_str, use_bids=use_bids)
    LGR.info(f"Saved sparse component to {coef_fn}")

    # Save low-rank component
    lowrank_path = os.path.join(out_dir, lowrank_fn)
    write_data(model.low_rank_, lowrank_path, masker, args.data[0], command_str, use_bids=use_bids)
    LGR.info(f"Saved low-rank component to {lowrank_fn}")

    # Save lambda values
    lambda_path = os.path.join(out_dir, lambda_fn)
    write_data(
        model.lambda_.reshape(1, -1),
        lambda_path,
        masker,
        args.data[0],
        command_str,
        use_bids=use_bids,
    )
    LGR.info(f"Saved lambda values to {lambda_fn}")


def _run_stability(args, command_str, out_dir):
    """Run stability selection workflow.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    command_str : str
        The command string for logging.
    out_dir : str
        Output directory.
    """
    from pySPFM.decomposition import StabilitySelection

    # Handle echo times
    te = args.te if args.te is not None else [0]

    # Read data
    data, masker = read_data(args.data[0], args.mask)

    # Create and fit the estimator
    model = StabilitySelection(
        tr=args.tr,
        te=te,
        hrf_model=args.hrf_model,
        block_model=args.block_model,
        n_surrogates=args.n_surrogates,
        n_lambdas=args.n_lambdas,
        threshold=args.threshold,
        n_jobs=args.n_jobs,
    )

    LGR.info("Fitting stability selection model...")
    model.fit(data)

    # Generate output filenames
    prefix = args.out_prefix
    use_bids = args.use_bids

    if use_bids:
        auc_fn = f"{prefix}_desc-AUC.nii.gz"
    else:
        auc_fn = f"{prefix}_pySPFM_AUC.nii.gz"

    # Save selection frequencies (AUC)
    auc_path = os.path.join(out_dir, auc_fn)
    write_data(
        model.selection_frequency_, auc_path, masker, args.data[0], command_str, use_bids=use_bids
    )
    LGR.info(f"Saved selection frequencies (AUC) to {auc_fn}")


def main(argv=None):
    """Entry point for the pySPFM CLI.

    Parameters
    ----------
    argv : list, optional
        Command-line arguments. If None, uses sys.argv.
    """
    if argv is None:
        argv = sys.argv

    parser = _get_parser()
    args = parser.parse_args(argv[1:])

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Set up output directory
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Set up logging
    debug = args.debug
    quiet = not (args.debug or args.verbose)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    log_fn = os.path.join(out_dir, f"pySPFM_{timestamp}.tsv")
    setup_loggers(logname=log_fn, quiet=quiet, debug=debug)

    # Save call script
    _save_call_script(out_dir, argv)

    # Build command string for AFNI history
    command_str = " ".join(argv)

    LGR.info(f"pySPFM version: {__version__}")
    LGR.info(f"Command: {command_str}")
    LGR.info(f"Output directory: {out_dir}")

    # Run the appropriate workflow
    if args.command == "sparse":
        _run_sparse(args, command_str, out_dir)
    elif args.command == "lowrank":
        _run_lowrank(args, command_str, out_dir)
    elif args.command == "stability":
        _run_stability(args, command_str, out_dir)

    # Save references
    refs_fn = os.path.join(out_dir, "_references.txt")
    with open(refs_fn, "w") as f:
        f.write("pySPFM references:\n\n")
        f.write("Please cite the following papers when using pySPFM:\n\n")
        f.write(
            "Caballero-Gaudes, C., et al. (2013). Paradigm Free Mapping with "
            "Sparse Regression Automatically Detects Single-Trial Functional "
            "Magnetic Resonance Imaging Blood Oxygenation Level Dependent "
            "Responses. Human Brain Mapping.\n"
        )
    LGR.info(f"Saved references to {refs_fn}")

    LGR.info("pySPFM completed successfully.")


if __name__ == "__main__":
    main()
