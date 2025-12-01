"""Stability Selection for deconvolution."""

import logging
import os

import jax
import jax.numpy as jnp
import numpy as np

from pySPFM._solvers.lars import solve_regularization_path

LGR = logging.getLogger("GENERAL")


def get_subsampling_indices(n_scans, n_echos, mode="same"):
    """
    Get subsampling indices to generate surrogate data for stability selection.

    Parameters
    ----------
    n_scans : int
        Number of scans.
    n_echos : int
        Number of echos.
    mode : str, optional
        Mode of subsampling. For multi-echo data you can choose between "same" and "different"
        subsampling indices across echos. The default is "same".

    Returns
    -------
    subsample_idx : np.ndarray
        Subsampling indices.
    """
    if "mode" in os.environ:  # only for testing
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
    """Calculate the AUC for a TR.

    Parameters
    ----------
    coefs : np.ndarray
        Matrix of coefficients of shape (n_lambdas, n_surrogates).
    lambdas : np.ndarray
        Array of lambdas of shape (n_lambdas).
    n_surrogates : int
        Number of surrogates.

    Returns
    -------
    auc : float
        AUC for a TR.
    """
    # Sum of all lambdas
    lambdas_sum = jnp.sum(lambdas)

    # Binarize coefficients
    coefs = jnp.where(coefs != 0, 1, 0)
    sum_ = 0

    # If coefs is two-dimensional, use the first dimension to calculate the AUC
    if coefs.ndim == 2:
        probs = jnp.sum(coefs, axis=1) / n_surrogates
        for i in range(len(lambdas)):
            sum_ += probs[i] * lambdas[i] / lambdas_sum

    # If coefs is one-dimensional, use the whole array to calculate the AUC
    elif coefs.ndim == 1:
        for i in range(len(lambdas)):
            sum_ += coefs[i] * lambdas[i] / lambdas_sum

    return sum_


def _get_tr_lambdas(estimates, lambdas, n_lambdas, n_surrogates):
    """Get lambdas and coefficients for a TR.

    Parameters
    ----------
    estimates : np.ndarray
        Matrix of coefficients of shape (n_lambdas, n_surrogates).
    lambdas : np.ndarray
        Array of lambdas of shape (n_lambdas, n_surrogates).
    n_lambdas : int
        Number of lambdas.
    n_surrogates : int
        Number of surrogates.

    Returns
    -------
    estimates_tr : np.ndarray
        Coefficients for a TR.
    lambdas_tr : np.ndarray
        Lambdas for a TR.
    """
    # Check if all lambdas are the same across axis 1.
    # If they are not, merge all lambdas into single, shared space.
    if not jnp.allclose(jnp.sum(lambdas, axis=1), n_lambdas * lambdas[:, 0]):
        estimates_tr, lambdas_tr = _generate_shared_lambdas_space(
            estimates, lambdas, n_lambdas, n_surrogates
        )
    else:
        estimates_tr = estimates
        lambdas_tr = lambdas[:, 0]

    return estimates_tr, lambdas_tr


def _generate_shared_lambdas_space(coefs, lambdas, n_lambdas, n_surrogates):
    """Generate shared space of lambdas and coefficients.

    Parameters
    ----------
    coefs : np.ndarray
        Coefficients of shape (n_lambdas, n_surrogates).
    lambdas : np.ndarray
        Lambdas of shape (n_lambdas, n_surrogates).
    n_lambdas : int
        Number of lambdas.
    n_surrogates : int
        Number of surrogates.

    Returns
    -------
    coefs_sorted : np.ndarray
        Sorted coefficients.
    lambdas_sorted : np.ndarray
        Sorted lambdas.
    """
    # Create shared space of lambdas and coefficients
    lambdas_shared = lambdas.reshape(n_lambdas * n_surrogates)
    coefs_shared = np.zeros(coefs.shape[0] * coefs.shape[1])

    # Project lambdas and coefficients into shared space
    for i in range(lambdas.shape[0]):
        coefs_shared[i * coefs.shape[1] : (i + 1) * coefs.shape[1]] = np.squeeze(coefs[i, :])

    # Sort lambdas and get the indices
    lambdas_sorted_idx = np.argsort(-lambdas_shared)
    lambdas_sorted = -np.sort(-lambdas_shared)

    # Sort coefficients
    coefs_sorted = coefs_shared[lambdas_sorted_idx]

    return coefs_sorted, lambdas_sorted


def stability_selection(hrf_norm, data, n_lambdas, n_surrogates, use_fista=False):
    """Stability Selection for deconvolution.

    Parameters
    ----------
    hrf_norm : np.ndarray
        Normalized HRF.
    data : np.ndarray
        Data.
    n_lambdas : int
        Number of lambdas.
    n_surrogates : int
        Number of surrogates.
    use_fista : bool, optional
        Whether to use FISTA in favor of LARS to solve the regularization path.

    Returns
    -------
    auc : np.ndarray
        AUC for each TR.
    """
    # Get n_scans, n_echos, n_voxels
    n_scans = hrf_norm.shape[1]
    n_echos = int(np.ceil(hrf_norm.shape[0] / n_scans))

    # Initialize variables to store the results
    estimates = np.zeros((n_scans, n_lambdas, n_surrogates))
    lambdas = np.zeros((n_lambdas, n_surrogates))

    # Generate surrogates and compute the regularization path
    stability_estimates = []
    for _ in range(n_surrogates):
        # Subsampling for Stability Selection
        subsample_idx = get_subsampling_indices(n_scans, n_echos)

        # Solve LARS
        fut_stability = solve_regularization_path(
            hrf_norm[subsample_idx, :], data[subsample_idx], n_lambdas, "stability", use_fista
        )
        stability_estimates.append(fut_stability)

    for surr_idx in range(n_surrogates):
        estimates[:, :, surr_idx] = np.squeeze(stability_estimates[surr_idx][0])
        lambdas[:, surr_idx] = np.squeeze(stability_estimates[surr_idx][1])

    # Calculate the AUC for each TR
    auc = np.zeros(n_scans)
    calculate_auc_jit = jax.jit(calculate_auc)
    for tr_idx in range(n_scans):
        # Get lambdas and coefficients for the TR
        estimates_tr, lambdas_tr = _get_tr_lambdas(
            estimates[tr_idx, :, :], lambdas, n_lambdas, n_surrogates
        )

        # Calculate AUC
        auc[tr_idx] = calculate_auc_jit(estimates_tr, lambdas_tr, n_surrogates).block_until_ready()

    return auc
