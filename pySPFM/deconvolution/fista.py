"""FISTA solver for PFM."""
import logging

import numpy as np
import pylops
from pyproximal.optimization.primal import AcceleratedProximalGradient
from pyproximal.proximal import L1, L2, L21_plus_L1
from scipy import linalg

from pySPFM.deconvolution.select_lambda import select_lambda

LGR = logging.getLogger("GENERAL")


def proximal_operator_lasso(y, thr):
    """Perform soft-thresholding.

    Parameters
    ----------
    y : ndarray
        Input data to be soft-thresholded.
    thr : float
        Thresholding value.

    Returns
    -------
    x : ndarray
        Soft-thresholded data.
    """
    x = y * np.maximum(np.zeros(y.shape), 1 - (thr / abs(y)))
    x[np.abs(x) < np.finfo(float).eps] = 0

    return x


def proximal_operator_mixed_norm(y, thr, rho_val=0.8, groups="space"):
    """Apply proximal operator for L2,1 + L1 mixed-norm.

    Parameters
    ----------
    y : ndarray
        Input data to be soft-thresholded.
    thr : float
        Thresholding value.
    rho_val : float, optional
        Weight for sparsity over grouping effect, by default 0.8
    groups : str, optional
        Dimension to apply grouping on, by default "space"

    Returns
    -------
    x : ndarray
        Data thresholded with L2,1 + L1 mixed-norm proximal operator.
    """
    # Division parameter of proximal operator
    div = np.nan_to_num(y / np.abs(y))

    # First parameter of proximal operator
    p_one = np.maximum(np.zeros(y.shape), (np.abs(y) - thr * rho_val))

    # Second parameter of proximal operator
    if groups == "space":
        foo = np.sum(np.maximum(np.zeros(y.shape), np.abs(y) - thr * rho_val) ** 2, axis=1)
        foo = foo.reshape(len(foo), 1)
        foo = np.dot(foo, np.ones((1, y.shape[1])))
    else:
        foo = np.sum(np.maximum(np.zeros(y.shape), np.abs(y) - thr * rho_val) ** 2, axis=0)
        foo = foo.reshape(1, len(foo))
        foo = np.dot(np.ones((y.shape[0], 1)), foo)

    p_two = np.maximum(
        np.zeros(y.shape),
        np.ones(y.shape) - np.nan_to_num(thr * (1 - rho_val) / np.sqrt(foo)),
    )

    # Proximal operation
    x = div * p_one * p_two

    # Return result
    return x


def fista(
    hrf,
    y,
    criterion="ut",
    lambda_=None,
    max_iter=400,
    min_iter=10,
    tol=1e-6,
    group=0.2,
    pcg=0.8,
    factor=10,
    lambda_echo=-1,
    use_pylops=False,
):
    """FISTA solver for PFM.

    Parameters
    ----------
    hrf : ndarray
        HRF matrix.
    y : ndarray
        Data to be deconvolved.
    criterion : str, optional
        Criterion to select regularization parameter lambda, by default "ut"
    lambda_ : float, optional
        Regularization parameter, by default None
    max_iter : int, optional
        Maximum number of iterations for FISTA, by default 400
    min_iter : int, optional
        Minimum number of iterations for FISTA, by default 10
    tol : float, optional
        Tolerance for residuals to find convergence of inverse problem, by default 1e-6
    group : float, optional
        Grouping (l2,1-norm) regularization parameter, by default 0.2
    pcg : float, optional
        Percentage of the maximum lambda possible to use as lambda, by default 0.8
    factor : int, optional
        Factor of the estimate of the level of noise to use as lambda, by default 10
    lambda_echo : int, optional
        When using multi-echo data, the number of TE to use to estimate the level of the noise,
        by default -1
    use_pylops : bool, optional
        Use pylops library to solve FISTA instead of using pySPFM's FISTA, by default False

    Returns
    -------
    S : ndarray
        Estimates of the activity-inducing (spike model) or innovation (block model) signal
    lambda_ : float
        Selected regularization parameter lambda
    """
    if len(y.shape) == 1:
        n_voxels = 1
        y = y[:, np.newaxis]
    else:
        n_voxels = y.shape[1]
    n_scans = hrf.shape[1]

    # Select lambda
    if lambda_ is None:
        lambda_, update_lambda, noise_estimate = select_lambda(
            hrf, y, criterion, factor, pcg, lambda_echo
        )
    else:
        update_lambda = False
        noise_estimate = 0

    c_ist = 1 / (linalg.norm(hrf) ** 2)

    if use_pylops:
        # Use pylops if lambda does not need to be updated
        hrf = pylops.MatrixMult(hrf)

        # Data fitting term
        l2 = L2(Op=hrf, b=y, densesolver="numpy")

        # Lambda and proximal operator
        if group == 0:
            prox = L1(sigma=lambda_)
        else:
            prox = L21_plus_L1(sigma=lambda_, rho=(1 - group))

        LGR.info("Performing FISTA with pylops...")

        S = AcceleratedProximalGradient(
            l2,
            prox,
            tau=c_ist,
            x0=np.zeros((n_scans, n_voxels)),
            epsg=np.ones(n_voxels),
            niter=max_iter,
            acceleration="fista",
            show=False,
        )
    else:
        # Use FISTA with updating lambda
        hrf_trans = hrf.T
        hrf_cov = np.dot(hrf_trans, hrf)
        v = np.dot(hrf_trans, y)

        y_fista_S = np.zeros((n_scans, n_voxels), dtype=np.float32)
        S = y_fista_S.copy()

        t_fista = 1

        precision = noise_estimate / 100000

        # Perform FISTA
        for num_iter in range(max_iter):

            # Save results from previous iteration
            S_old = S.copy()
            y_ista_S = y_fista_S.copy()

            S_fidelity = v - np.dot(hrf_cov, y_ista_S)

            # Forward-Backward step
            z_ista_S = y_ista_S + c_ist * S_fidelity

            # Estimate S
            if group > 0:
                S = proximal_operator_mixed_norm(z_ista_S, c_ist * lambda_, rho_val=(1 - group))
            else:
                S = proximal_operator_lasso(z_ista_S, c_ist * lambda_)

            t_fista_old = t_fista
            t_fista = 0.5 * (1 + np.sqrt(1 + 4 * (t_fista_old**2)))

            y_fista_S = S + (S - S_old) * (t_fista_old - 1) / t_fista

            # Convergence
            if num_iter >= min_iter:
                nonzero_idxs_rows, nonzero_idxs_cols = np.where(
                    np.abs(S) > 10 * np.finfo(float).eps
                )
                diff = np.abs(
                    S[nonzero_idxs_rows, nonzero_idxs_cols]
                    - S_old[nonzero_idxs_rows, nonzero_idxs_cols]
                )
                convergence_criteria = np.abs(diff / S_old[nonzero_idxs_rows, nonzero_idxs_cols])

                if np.all(convergence_criteria <= tol):
                    break

            LGR.debug(f"Iteration: {str(num_iter)} / {str(max_iter)}")

            # Update lambda
            if update_lambda:
                nv = np.sqrt(np.sum((np.dot(hrf, S) - y) ** 2, axis=0) / n_scans)
                if abs(nv - noise_estimate) > precision:
                    lambda_ = np.nan_to_num(lambda_ * noise_estimate / nv)

    return S, lambda_
