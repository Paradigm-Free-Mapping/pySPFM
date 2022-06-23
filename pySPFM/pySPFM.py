import datetime
import logging
import os
import sys
from os import path as op

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from pySPFM import utils
from pySPFM.cli.run import _get_parser
from pySPFM.deconvolution.debiasing import debiasing_block, debiasing_spike
from pySPFM.deconvolution.fista import fista
from pySPFM.deconvolution.hrf_generator import HRFMatrix
from pySPFM.deconvolution.lars import solve_regularization_path
from pySPFM.deconvolution.select_lambda import select_lambda
from pySPFM.deconvolution.spatial_regularization import spatial_tikhonov
from pySPFM.io import read_data, write_data

LGR = logging.getLogger("GENERAL")
RefLGR = logging.getLogger("REFERENCES")


def pySPFM(
    data_fn,
    mask_fn,
    output_filename,
    tr,
    out_dir,
    te=[0],
    block_model=False,
    hrf_model="spm",
    debias=True,
    group=0.2,
    criterion="bic",
    pcg=0.8,
    factor=10,
    lambda_echo=-1,
    max_iter_factor=1.0,
    max_iter_fista=500,
    max_iter_spatial=100,
    max_iter=10,
    min_iter_fista=50,
    n_jobs=1,
    spatial_weight=0,
    spatial_lambda=1,
    spatial_dim=3,
    mu=0.01,
    tolerance=1e-6,
    is_atlas=False,
    debug=False,
    quiet=False,
):
    data_str = str(data_fn).strip("[]")
    te_str = str(te).strip("[]")
    arguments = f"-i {data_str} -m {mask_fn} -o {output_filename} -tr {tr} "
    arguments += f"-d {out_dir} -te {te_str} -group {group} -crit {criterion} "

    if block_model:
        arguments += "-block "
    if debug:
        arguments += "-debug "
    if quiet:
        arguments += "-quiet"
    command_str = f"pySPFM {arguments}"

    # Generate output directory if it doesn't exist
    out_dir = op.abspath(out_dir)
    if not op.isdir(out_dir):
        os.mkdir(out_dir)

    # Save command into sh file
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
        nvoxels = data_masked.shape[1]
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
                nvoxels = data_temp.shape[1]
            else:
                # data_temp, _, _, _ = read_data(data_fn[te_idx], mask_fn, mask_idxs)
                data_masked = np.concatenate((data_masked, data_temp), axis=0)

            LGR.info(f"{te_idx + 1}/{n_te} echoes...")

    LGR.info("Data read.")

    # Generate design matrix with shifted versions of HRF
    LGR.info("Generating design matrix with shifted versions of HRF...")
    hrf_obj = HRFMatrix(te=te, block=block_model, model=hrf_model)
    hrf = hrf_obj.generate_hrf(tr=tr, n_scans=n_scans).hrf_

    # Run LARS if bic or aic on given.
    # If another criterion is given, then solve with FISTA.
    lars_criteria = ["bic", "aic"]
    fista_criteria = ["mad", "mad_update", "ut", "lut", "factor", "pcg", "eigval"]

    # Run for loop only once for just temporal regularization
    if spatial_weight == 0:
        max_iter = 1
    else:
        # Initialize variables for spatial regularization
        estimates_temporal = np.empty((n_scans, nvoxels))
        estimates_spatial = np.empty((n_scans, nvoxels))
        final_estimates = np.empty((n_scans, nvoxels))

    # Iterate between temporal and spatial regularizations
    LGR.info("Solving inverse problem...")
    for iter_idx in range(max_iter):
        if spatial_weight > 0:
            data_temp_reg = final_estimates - estimates_temporal + data_masked
        else:
            data_temp_reg = data_masked

        estimates = np.zeros((n_scans, nvoxels))
        lambda_map = np.zeros(nvoxels)

        if criterion in lars_criteria:
            nlambdas = max_iter_factor * n_scans
            # Solve LARS for each voxel with parallelization
            lars_estimates = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
                delayed(solve_regularization_path)(
                    hrf, data_temp_reg[:, vox_idx], nlambdas, criterion
                )
                for vox_idx in tqdm(range(nvoxels))
            )

            for vox_idx in range(nvoxels):
                estimates[:, vox_idx] = np.squeeze(lars_estimates[vox_idx][0])
                lambda_map[vox_idx] = np.squeeze(lars_estimates[vox_idx][1])

        elif criterion in fista_criteria:
            # Solve fista
            fista_estimates = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
                delayed(fista)(
                    hrf,
                    data_temp_reg[:, vox_idx],
                    criterion,
                    max_iter_fista,
                    min_iter_fista,
                    tolerance,
                    group,
                    pcg,
                    factor,
                    lambda_echo,
                )
                for vox_idx in tqdm(range(nvoxels))
            )
            for vox_idx in range(nvoxels):
                estimates[:, vox_idx] = np.squeeze(fista_estimates[vox_idx][0])
                lambda_map[vox_idx] = np.squeeze(fista_estimates[vox_idx][1])

        else:
            raise ValueError("Wrong criterion option given.")

        # Convolve with HRF
        if block_model:
            estimates_block = estimates
            hrf_obj = HRFMatrix(te=te, block=False, model=hrf_model)
            hrf = hrf_obj.generate_hrf(tr=tr, n_scans=n_scans).hrf_
            estimates_spike = np.dot(np.tril(np.ones(n_scans)), estimates_block)
            fitts = np.dot(hrf, estimates_spike)
        else:
            estimates_spike = estimates
            fitts = np.dot(hrf, estimates_spike)

        # Perform spatial regularization if a weight is given
        if spatial_weight > 0:
            # Update temporal estimates
            estimates_temporal = estimates_temporal + (estimates - final_estimates)

            # Calculates for the whole volume
            estimates_tikhonov = spatial_tikhonov(
                final_estimates,
                final_estimates - estimates_spatial + data_masked,
                mask,
                max_iter_spatial,
                spatial_dim,
                spatial_lambda,
                mu,
            )

            # Update spatial estimates
            estimates_spatial = estimates_spatial + (estimates_tikhonov - final_estimates)

            # Calculate final estimates
            final_estimates = (
                estimates_temporal * (1 - spatial_weight) + spatial_weight * estimates_spatial
            )
        else:
            final_estimates = estimates

    LGR.info("Inverse problem solved.")

    # Perform debiasing step
    if debias:
        LGR.info("Debiasing estimates...")
        if block_model:
            hrf_obj = HRFMatrix(te=te, block=False, model=hrf_model)
            hrf = hrf_obj.generate_hrf(tr=tr, n_scans=n_scans).hrf_
            estimates_spike = debiasing_block(
                hrf=hrf, y=data_masked, estimates_matrix=final_estimates, jobs=n_jobs
            )
            fitts = np.dot(hrf, estimates_spike)
        else:
            estimates_spike, fitts = debiasing_spike(
                hrf, data_masked, final_estimates, jobs=n_jobs
            )

    LGR.info("Saving results...")
    # Save innovation signal
    if block_model:
        estimates_block = final_estimates
        output_name = f"{output_filename}_innovation.nii.gz"
        write_data(
            estimates_block,
            os.path.join(out_dir, output_name),
            mask,
            data_header,
            command_str,
            is_atlas=is_atlas,
        )

        if not debias:
            hrf_obj = HRFMatrix(te=te, block=False, model=hrf_model)
            hrf = hrf_obj.generate_hrf(tr=tr, n_scans=n_scans).hrf_
            estimates_spike = np.dot(np.tril(np.ones(n_scans)), estimates_block)
            fitts = np.dot(hrf, estimates_spike)

    # Save activity-inducing signal
    if n_te == 1:
        output_name = f"{output_filename}_beta.nii.gz"
    elif n_te > 1:
        output_name = f"{output_filename}_DR2.nii.gz"
    write_data(
        estimates_spike,
        os.path.join(out_dir, output_name),
        mask,
        data_header,
        command_str,
        is_atlas=is_atlas,
    )

    # Save fitts
    if n_te == 1:
        output_name = f"{output_filename}_fitts.nii.gz"
        write_data(
            fitts,
            os.path.join(out_dir, output_name),
            mask,
            data_header,
            command_str,
            is_atlas=is_atlas,
        )
    elif n_te > 1:
        for te_idx in range(n_te):
            te_data = fitts[te_idx * n_scans : (te_idx + 1) * n_scans, :]
            output_name = f"{output_filename}_dr2HRF_E0{te_idx + 1}.nii.gz"
            write_data(
                te_data,
                os.path.join(out_dir, output_name),
                mask,
                data_header,
                command_str,
                is_atlas=is_atlas,
            )

    # Save noise estimate
    for te_idx in range(n_te):
        output_name = f"{output_filename}_MAD_E0{te_idx + 1}.nii.gz"
        if te_idx == 0:
            y_echo = data_masked[:n_scans, :]
        else:
            y_echo = data_masked[te_idx * n_scans : (te_idx + 1) * n_scans, :]
        _, _, noise_estimate = select_lambda(hrf=hrf, y=y_echo)
        write_data(
            np.expand_dims(noise_estimate, axis=0),
            os.path.join(out_dir, output_name),
            mask,
            data_header,
            command_str,
            is_atlas=is_atlas,
        )

    # Save lambda
    output_name = f"{output_filename}_lambda.nii.gz"
    write_data(
        np.expand_dims(lambda_map, axis=0),
        os.path.join(out_dir, output_name),
        mask,
        data_header,
        command_str,
        is_atlas=is_atlas,
    )

    LGR.info("Results saved.")

    LGR.info("pySPFM finished.")
    utils.teardown_loggers()


def _main(argv=None):
    """pySPFM entry point"""
    options = _get_parser().parse_args(argv)
    pySPFM(**vars(options))


if __name__ == "__main__":
    _main(sys.argv[1:])
