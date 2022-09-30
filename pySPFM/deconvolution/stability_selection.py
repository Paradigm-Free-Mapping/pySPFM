import logging
import os

import numpy as np

from pySPFM.deconvolution.lars import solve_regularization_path

LGR = logging.getLogger("GENERAL")


def get_subsampling_indices(n_scans, n_echos, mode="same"):
    if "mode" in os.environ.keys():  # only for testing
        np.random.seed(200)
    # Subsampling for Stability Selection
    if mode == "different":  # different time points are selected across echoes
        subsample_idx = np.sort(
            np.random.choice(range(n_scans), int(0.6 * n_scans), 0)
        )  # 60% of timepoints are kept
        for i in range(n_echos - 1):
            subsample_idx = np.concatenate(
                (
                    subsample_idx,
                    np.sort(
                        np.random.choice(
                            range((i + 1) * n_scans, (i + 2) * n_scans),
                            int(0.6 * n_scans),
                            0,
                        )
                    ),
                )
            )
    elif mode == "same":  # same time points are selected across echoes
        subsample_idx = np.sort(
            np.random.choice(range(n_scans), int(0.6 * n_scans), 0)
        )  # 60% of timepoints are kept

    return subsample_idx


def calculate_auc(coefs, lambdas, n_surrogates):

    # Create shared space of lambdas and coefficients
    lambdas_shared = np.zeros((lambdas.shape[0] * lambdas.shape[1]))
    coefs_shared = np.zeros((coefs.shape[0] * coefs.shape[1]))

    # Project lambdas and coefficients into shared space
    for i in range(lambdas.shape[0]):
        lambdas_shared[i * lambdas.shape[1] : (i + 1) * lambdas.shape[1]] = np.squeeze(
            lambdas[i, :]
        )
        coefs_shared[i * coefs.shape[1] : (i + 1) * coefs.shape[1]] = np.squeeze(coefs[i, :])

    # Sort lambdas and get the indices
    lambdas_sorted_idx = np.argsort(lambdas_shared)
    lambdas_sorted = np.sort(lambdas_shared)

    # Sort coefficients to match lambdas
    coefs_sorted = coefs_shared[lambdas_sorted_idx]

    # Turn coefs_sorted into a binary vector
    coefs_sorted[coefs_sorted != 0] = 1

    # Calculate the AUC as the normalized area under the curve
    auc = np.trapz(coefs_sorted, lambdas_sorted) / np.sum(lambdas_sorted) / n_surrogates

    return auc


def stability_selection(hrf_norm, data, n_lambdas, n_surrogates):
    # Get n_scans, n_echos, n_voxels
    n_scans = hrf_norm.shape[1]
    n_echos = int(np.ceil(hrf_norm.shape[0] / n_scans))

    # Initialize variables to store the results
    estimates = np.zeros((n_scans, n_lambdas, n_surrogates))
    lambdas = np.zeros((n_lambdas, n_surrogates))

    # Generate surrogates and compute the regularization path
    stability_estimates = []
    for surr_idx in range(n_surrogates):
        # Subsampling for Stability Selection
        subsample_idx = get_subsampling_indices(n_scans, n_echos)

        # Solve LARS
        fut_stability = solve_regularization_path(
            hrf_norm[subsample_idx, :], data[subsample_idx], n_lambdas, "stability"
        )
        stability_estimates.append(fut_stability)

    for surr_idx in range(n_surrogates):
        estimates[:, :, surr_idx] = np.squeeze(stability_estimates[surr_idx][0])
        lambdas[:, surr_idx] = np.squeeze(stability_estimates[surr_idx][1])

    # Calculate the AUC for each TR
    auc = np.zeros((n_scans))
    for tr_idx in range(n_scans):
        auc[tr_idx] = calculate_auc(estimates[tr_idx, :, :], lambdas, n_surrogates)

    return auc
