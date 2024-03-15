"""FISTA solver for PFM."""

import logging

import jax
import jax.numpy as jnp
import numpy as np
import pylops
from pyproximal.optimization.primal import AcceleratedProximalGradient
from pyproximal.proximal import L1, L2, L21_plus_L1

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
    x = y * jnp.maximum(jnp.zeros(y.shape), 1 - (thr / jnp.abs(y)))

    # Set values to zero if they are smaller than 1e-10
    x = jnp.where(jnp.abs(x) < 1e-10, jnp.zeros(x.shape), x)

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
    div = jnp.nan_to_num(y / jnp.abs(y))

    # First parameter of proximal operator
    p_one = jnp.maximum(jnp.zeros(y.shape), (jnp.abs(y) - thr * rho_val))

    # Second parameter of proximal operator
    if groups == "space":
        foo = jnp.sum(jnp.maximum(jnp.zeros(y.shape), jnp.abs(y) - thr * rho_val) ** 2, axis=1)
        foo = foo.reshape(len(foo), 1)
        foo = jnp.dot(foo, jnp.ones((1, y.shape[1])))
    else:
        foo = jnp.sum(jnp.maximum(np.zeros(y.shape), jnp.abs(y) - thr * rho_val) ** 2, axis=0)
        foo = foo.reshape(1, len(foo))
        foo = jnp.dot(jnp.ones((y.shape[0], 1)), foo)

    p_two = jnp.maximum(
        jnp.zeros(y.shape),
        jnp.ones(y.shape) - jnp.nan_to_num(thr * (1 - rho_val) / jnp.sqrt(foo)),
    )

    return div * p_one * p_two


def _fista_forward(v, hrf_cov, y_ista_S, c_ist):
    """Forward step of FISTA.

    Parameters
    ----------
    v : ndarray
        Covariance matrix of the data and the HRF
    hrf_cov : ndarray
        Covariance matrix of the HRF
    y_ista_S : ndarray
        FISTA value of the current iteration
    c_ist : float
        ISTA parameter

    Returns
    -------
    ndarray
        Updated estimate of the activity-inducing (spike model) or innovation (block model) signal
    """

    S_fidelity = v - jnp.dot(hrf_cov, y_ista_S)

    return y_ista_S + c_ist * S_fidelity


def _fista_update(t_fista, S, S_old):
    """Update FISTA parameters.

    Parameters
    ----------
    t_fista : float
        Current FISTA parameter
    S : ndarray
        Current estimate of the activity-inducing (spike model) or innovation (block model) signal
    S_old : ndarray
        Previous estimate of the activity-inducing (spike model) or innovation (block model) signal

    Returns
    -------
    t_fista : float
        Current FISTA parameter
    y_fista_S : ndarray
        Current estimate of the activity-inducing (spike model) or innovation (block model) signal
    """
    t_fista_old = t_fista
    t_fista = 0.5 * (1 + jnp.sqrt(1 + 4 * (t_fista_old**2)))

    y_fista_S = S + (S - S_old) * (t_fista_old - 1) / t_fista

    return t_fista, y_fista_S


def _has_converged(S, S_old, tol=1e-6):
    """Check if FISTA has converged.

    Parameters
    ----------
    S : ndarray
        Current estimate of the activity-inducing (spike model) or innovation (block model) signal
    S_old : ndarray
        Previous estimate of the activity-inducing (spike model) or innovation (block model) signal
    tol : float, optional
        Tolerance for residuals to find convergence of inverse problem, by default 1e-6

    Returns
    -------
    bool
        True if FISTA has converged, False otherwise
    """

    # Calculate normalized error between current and previous estimate
    estimate_error = jnp.abs(S - S_old) / jnp.abs(S_old)

    # Check if the error is smaller than the tolerance for all voxels
    return jnp.all(jnp.abs(estimate_error) <= tol).astype(jnp.bool_)


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
    positive_only=False,
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
    positive_only : bool, optional
        If True, the estimated signal will be forced to be positive, by default False

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

    c_ist = 1 / (jnp.linalg.norm(hrf) ** 2)

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

        if positive_only:
            S = jnp.maximum(np.sign(hrf[0, 0]) * S, 0)

    else:
        # Use FISTA with updating lambda
        hrf_trans = hrf.T
        hrf_cov = jnp.dot(hrf_trans, hrf)
        v = jnp.dot(hrf_trans, y)

        y_fista_S = jnp.zeros((n_scans, n_voxels), dtype=jnp.float32)
        S = y_fista_S.copy()

        t_fista = 1

        precision = noise_estimate / 100000

        # Compile jit functions
        if group > 0:
            proximal_operator_mixed_norm_jit = jax.jit(proximal_operator_mixed_norm)
        else:
            proximal_operator_lasso_jit = jax.jit(proximal_operator_lasso)

        _fista_forward_jit = jax.jit(_fista_forward)
        _fista_update_jit = jax.jit(_fista_update)
        _has_converged_jit = jax.jit(_has_converged)

        # Perform FISTA
        for num_iter in range(max_iter):
            # Save results from previous iteration
            S_old = S.copy()
            y_ista_S = y_fista_S.copy()

            z_ista_S = _fista_forward_jit(v, hrf_cov, y_ista_S, c_ist).block_until_ready()

            # Estimate S
            if group > 0:
                S = proximal_operator_mixed_norm_jit(
                    z_ista_S, c_ist * lambda_, rho_val=(1 - group)
                ).block_until_ready()
            else:
                S = proximal_operator_lasso_jit(z_ista_S, c_ist * lambda_).block_until_ready()

            if positive_only:
                S = jnp.maximum(np.sign(hrf[0, 0]) * S, 0)

            t_fista, y_fista_S = _fista_update_jit(t_fista, S, S_old)

            # Convergence
            if num_iter >= min_iter and _has_converged_jit(S_old, S, tol).block_until_ready():
                break

            LGR.debug(f"Iteration: {str(num_iter)} / {str(max_iter)}")

            # Update lambda
            if update_lambda:
                nv = np.sqrt(np.sum((np.dot(hrf, S) - y) ** 2, axis=0) / n_scans)
                if abs(nv - noise_estimate) > precision:
                    lambda_ = np.nan_to_num(lambda_ * noise_estimate / nv)

    return S, lambda_
