"""Debiasing functions for PFM."""

import logging

import numpy as np
import scipy as sci
from dask import compute
from dask import delayed as delayed_dask
from scipy.signal import find_peaks
from sklearn.linear_model import RidgeCV

from pySPFM.utils import dask_scheduler

LGR = logging.getLogger("GENERAL")
RefLGR = logging.getLogger("REFERENCES")


def group_hrf(hrf, non_zero_idxs, group_dist=3):
    """Group HRFs based on the distance between non-zero coefficients.

    Parameters
    ----------
    hrf : (E x T) ndarray
        Matrix containing shifted HRFs in its columns. E stands for the number of volumes times
        the number of echo-times.
    non_zero_idxs : (T) ndarray
        Array containing the indexes of the non-zero coefficients.
    group_dist : int, optional
        Maximum distance between non-zero coefficients to be considered as part of the same group,
        by default 3

    Returns
    -------
    hrf_out : (E x T) ndarray
        Matrix containing shifted HRFs in its columns. E stands for the number of volumes times
        the number of echo-times.
    new_idxs : (T) ndarray
        Array containing the indexes of the non-zero coefficients.
    """
    temp = np.zeros(hrf.shape[0])
    hrf_out = np.zeros(hrf.shape)
    non_zeros_flipped = np.flip(non_zero_idxs)
    new_idxs = []

    for iter_idx, non_zero_idx in enumerate(non_zeros_flipped):
        if (
            iter_idx != len(non_zeros_flipped) - 1
            and abs(non_zero_idx - non_zeros_flipped[iter_idx + 1]) <= group_dist
        ):
            temp += hrf[:, non_zero_idx]
        else:
            temp += hrf[:, non_zero_idx]
            hrf_out[:, non_zero_idx] = temp
            temp = np.zeros(hrf.shape[0])
            new_idxs.append(non_zero_idx)

    new_idxs = np.flip(new_idxs)
    hrf_out = hrf_out[:, new_idxs]

    return hrf_out, new_idxs


def group_betas(betas, non_zero_idxs, group_dist=3):
    """Group betas based on the distance between non-zero coefficients.

    Parameters
    ----------
    betas : (T) ndarray
        Array containing the non-zero coefficients selected as neuronal-related.
    non_zero_idxs : (T) ndarray
        Array containing the indexes of the non-zero coefficients.
    group_dist : int, optional
        Maximum distance between non-zero coefficients to be considered as part of the same group,
        by default 3

    Returns
    -------
    betas : (T) ndarray
        Array containing the non-zero coefficients selected as neuronal-related.
    """
    for i in range(len(non_zero_idxs)):
        if i > 0 and (non_zero_idxs[i] - non_zero_idxs[i - 1]) <= group_dist:
            betas[non_zero_idxs[i]] = betas[non_zero_idxs[i - 1]]

    return betas


# Performs the debiasing step on an estimates_matrix timeseries obtained considering the
# integrator model
def innovation_to_block(hrf, y, estimates_matrix, is_ls):
    """Perform debiasing with the block model.

    Parameters
    ----------
    hrf : (E x T) ndarray
        Matrix containing shifted HRFs in its columns. E stands for the number of volumes times
        the number of echo-times.
    y : (T x S) ndarray
        Matrix with fMRI data provided to pySPFM.
    estimates_matrix : (T x S) ndarray
        Matrix containing the non-zero coefficients selected as neuronal-related.
    is_ls : bool
        Whether least squares is solved in favor of ridge regression.

    Returns
    -------
    beta : (T x S) ndarray
        Debiased activity-inducing signal obtained from estimated innovation signal.
    s : (T x L) ndarray
        Transformation matrix used to integrate the innovation signal into activity-inducing
        signal. L stands for the number of steps to integrate.
    """
    # Find indexes of nonzero coefficients
    nonzero_idxs = np.where(estimates_matrix != 0)[0]
    n_nonzero = len(nonzero_idxs)  # Number of nonzero coefficients

    # Initiates beta
    beta = np.zeros((estimates_matrix.shape))
    s = 0

    if n_nonzero != 0:
        # Initiates matrix s and array of labels
        s = np.zeros((hrf.shape[1], n_nonzero + 1))
        labels = np.zeros((estimates_matrix.shape[0]))

        # Gives values to s design matrix based on nonzeros in estimates_matrix
        # It also stores the labels of the changes in the design matrix
        # to later generate the debiased timeseries with the obtained betas
        for idx in range(n_nonzero + 1):
            if idx == 0:
                s[0 : nonzero_idxs[idx], idx] = 1
                labels[: nonzero_idxs[idx]] = idx
            elif idx == n_nonzero:
                s[nonzero_idxs[idx - 1] :, idx] = 1
                labels[nonzero_idxs[idx - 1] :] = idx
            else:
                s[nonzero_idxs[idx - 1] : nonzero_idxs[idx], idx] = 1
                labels[nonzero_idxs[idx - 1] : nonzero_idxs[idx]] = idx

        # Performs the least squares to obtain the beta amplitudes
        if is_ls:
            beta_amplitudes, _, _, _ = np.linalg.lstsq(
                np.dot(hrf, s), y, rcond=None
            )  # b-ax --> returns x
        else:
            clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10]).fit(np.dot(hrf, s), y)
            beta_amplitudes = clf.coef_

        # Positions beta amplitudes in the entire timeseries
        for amp_change in range(n_nonzero + 1):
            beta[labels == amp_change] = beta_amplitudes[amp_change]

    return beta, s


def do_debias_block(hrf, y, estimates_matrix, dist=2):
    """Perform debiasing with the block model.

    Parameters
    ----------
    hrf : (E x T) ndarray
        Matrix containing shifted HRFs in its columns. E stands for the number of volumes times
        the number of echo-times.
    y : (T x 1) ndarray
        Matrix with fMRI data provided to pySPFM in a voxel.
    estimates_matrix : (T x 1) ndarray
        Matrix containing the non-zero coefficients selected as neuronal-related in a voxel.
    dist : int, optional
        Minimum number of TRs in between of the peaks found, by default 2

    Returns
    -------
    beta_out : ndarray
        Debiased activity-inducing signal obtained from estimated innovation signal in a voxel.
    """
    # Keep only maximum values in estimates_matrix peaks
    temp = np.zeros((estimates_matrix.shape[0],))
    peak_idxs, _ = find_peaks(abs(estimates_matrix), distance=dist)
    temp[peak_idxs] = estimates_matrix[peak_idxs].copy()

    estimates_matrix = temp.copy()

    beta_out, _ = innovation_to_block(hrf, y, estimates_matrix, is_ls=True)

    return beta_out


def debiasing_block(hrf, y, estimates_matrix, n_jobs=4, dist=2):
    """Voxelwise block model debiasing workflow.

    Parameters
    ----------
    hrf : (E x T) ndarray
        Matrix containing shifted HRFs in its columns. E stands for the number of volumes times
        the number of echo-times.
    y : (T x S) ndarray
        Matrix with fMRI data provided to pySPFM.
    estimates_matrix : (T x S) ndarray
        Matrix containing the non-zero coefficients selected as neuronal-related.
    n_jobs : int, optional
        Number of jobs to run in parallel, by default 4
    dist : int, optional
        Minimum number of TRs in between of the peaks found, by default 2

    Returns
    -------
    beta_out : ndarray
        Debiased activity-inducing signal obtained from estimated innovation signal.
    """
    n_scans = estimates_matrix.shape[0]
    n_voxels = estimates_matrix.shape[1]

    # Initiates beta matrix
    beta_out = np.zeros((n_scans, n_voxels))

    LGR.info("Starting debiasing step...")
    # Performs debiasing
    client, cluster = dask_scheduler(n_jobs)
    futures = []
    for voxidx in range(n_voxels):
        fut = delayed_dask(do_debias_block, pure=False)(
            hrf, y[:, voxidx], estimates_matrix[:, voxidx], dist
        )
        futures.append(fut)
    debiased = compute(futures)[0]

    # Close the client and cluster
    if client is not None:
        client.close()
        cluster.close()

    for vox_idx in range(n_voxels):
        beta_out[:, vox_idx] = debiased[vox_idx]

    LGR.info("Debiasing step finished")
    return beta_out


def do_debias_spike(hrf, y, estimates_matrix, group=False, group_dist=3, non_negative=False):
    """Perform debiasing with the spike model.

    Parameters
    ----------
    hrf : (E x T) ndarray
        Matrix containing shifted HRFs in its columns. E stands for the number of volumes times
        the number of echo-times.
    y : (T x 1) ndarray
        Array with fMRI data of a voxel provided to pySPFM.
    estimates_matrix : (T x 1) ndarray
        Array containing the non-zero coefficients selected as neuronal-related.
    group : bool, optional
        Whether to group the spikes or not, by default False
    group_dist : int, optional
        Minimum number of TRs in between of the peaks found, by default 3
    non_negative : bool, optional
        Whether to perform non-negative least squares or not, by default False

    Returns
    -------
    beta_out : ndarray
        Debiased activity-inducing signal in a voxel.
    fitts_out : ndarray
        Debiased activity-inducing signal convolved with the HRF in a voxel.
    """
    index_events_opt = np.where(abs(estimates_matrix) > 10 * np.finfo(float).eps)[0]
    beta2save = np.zeros((estimates_matrix.shape[0], 1))

    if group:
        hrf_events, index_events_opt_group = group_hrf(hrf, index_events_opt, group_dist)
    else:
        hrf_events = hrf[:, index_events_opt]

    if non_negative:
        # If the first value of the hrf is negative, then it's multi-echo data
        if hrf_events[0, 0] < 0:
            multi_echo = True
            hrf_events = -hrf_events
        coef_ls_fitdebias, _, _, _ = sci.optimize.nnls(hrf_events, y.T, cond=None)
        # Flip the sign back on the coefficients if multi-echo
        if multi_echo:
            coef_ls_fitdebias = -coef_ls_fitdebias

    else:
        coef_ls_fitdebias, _, _, _ = sci.linalg.lstsq(hrf_events, y.T, cond=None)

    if group:
        beta2save[index_events_opt_group, 0] = coef_ls_fitdebias
    else:
        beta2save[index_events_opt, 0] = coef_ls_fitdebias
    fitts_out = np.squeeze(np.dot(hrf, beta2save))

    beta_out = beta2save.reshape(len(beta2save))
    if group:
        beta_out = group_betas(beta_out, index_events_opt, group_dist)

    return beta_out, fitts_out


def debiasing_spike(
    hrf, y, estimates_matrix, n_jobs=0, group=False, group_dist=3, non_negative=False
):
    """Perform voxelwise debiasing with spike model.

    Parameters
    ----------
    hrf : (E x T) ndarray
        Matrix containing shifted HRFs in its columns. E stands for the number of volumes times
        the number of echo-times.
    y : (T x S) ndarray
        Matrix with fMRI data provided to pySPFM.
    estimates_matrix : (T x S) ndarray
        Matrix containing the non-zero coefficients selected as neuronal-related.
    n_jobs : int, optional
        Number of jobs to run in parallel, by default 0
    group : bool, optional
        Whether to group the spikes or not, by default False
    group_dist : int, optional
        Minimum number of TRs in between of the peaks found, by default 3
    non_negative : bool, optional
        Whether to perform non-negative least squares or not, by default False

    Returns
    -------
    beta_out : ndarray
        Debiased activity-inducing signal.
    fitts_out : ndarray
        Debiased activity-inducing signal convolved with the HRF.
    """
    beta_out = np.zeros(estimates_matrix.shape)
    fitts_out = np.zeros(y.shape)

    index_voxels = np.unique(np.where(abs(estimates_matrix) > 10 * np.finfo(float).eps)[1])

    LGR.info("Performing debiasing step...")
    client, cluster = dask_scheduler(n_jobs)
    futures = []
    for voxidx in range(len(index_voxels)):
        fut = delayed_dask(do_debias_spike, pure=False)(
            hrf,
            y[:, index_voxels[voxidx]],
            estimates_matrix[:, index_voxels[voxidx]],
            group,
            group_dist,
            non_negative,
        )
        futures.append(fut)
    debiased = compute(futures)[0]

    # Close the client and cluster
    if client is not None:
        client.close()
        cluster.close()

    for voxidx in range(len(index_voxels)):
        beta_out[:, index_voxels[voxidx]] = debiased[voxidx][0]
        fitts_out[:, index_voxels[voxidx]] = debiased[voxidx][1]

    LGR.info("Debiasing step finished")
    return beta_out, fitts_out
