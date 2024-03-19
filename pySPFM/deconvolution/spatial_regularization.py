"""Spatial regularization functions as developed in Total Activation."""

import numpy as np
from nilearn.masking import apply_mask, unmask


def spatial_tikhonov(estimates, data, masker, niter, dim, lambda_, mu):
    r"""Spatial regularization technique with Tikhonov regularization as in Total Activation.

    This function computes the tikhonov regularization

    :math:`\\mathbf{F(x)} = \\min \\| \\mathbf{y} - \\mathbf{x} \\|^2 + \\lambda \\cdot
    \\| \\Delta \\mathbf{x}\\|^2`.

    Delta is the laplacian operator delta[n] = [1 -2 1]; so symmetric, in
    matrix form :math:`\\Delta^T = \\Delta`.

    Parameters
    ----------
    estimates : ndarray
        Estimates (output of temporal regularization).
    data : ndarray
        Observations.
    masker : nilearn.maskers.NiftiMasker
        Masker image to unmask and mask estimates (2D to 4D and back).
    niter : int
        Number of iterations to perform spatial regularization.
    dim : int
        Slice-wise regularization with dim = 2; whole-volume regularization with dim=3.
        Default = 3.
    lambda_ : float
        Spatial regularization parameter. Default = 1.
    mu : float
        Step size (small, ~0.01)

    Returns
    -------
    final_estimates: ndarray
        Estimates of activity-inducing or innovation signal after spatial regularization.
    """
    # Transform data from 2D into 4D
    estimates_vol = masker.inverse_transform(estimates)
    data_vol = masker.inverse_trasnform(data)

    if dim == 2:
        h = generate_delta(dim=dim)

        h = np.fft.fft2(h, (estimates_vol.shape[0], estimates_vol.shape[1]))

        for iter_idx in range(niter):
            for time_idx in range(estimates_vol.shape[-1]):
                for slice_idx in range(estimates_vol.shape[2]):
                    estimates_vol[:, :, slice_idx, time_idx] = (
                        (1 - mu) * estimates_vol[:, :, slice_idx, time_idx]
                        + mu * data_vol[:, :, slice_idx, time_idx]
                        - mu
                        * lambda_
                        * np.fft.ifft2(
                            (np.conj(h) * h)
                            * np.fft.fft2(
                                estimates_vol[:, :, slice_idx, time_idx],
                                (estimates_vol.shape[0], estimates_vol.shape[1]),
                            )
                        )
                    )

    elif dim == 3:
        h = generate_delta(dim=dim)

        h = np.fft.fftn(h, estimates_vol.shape[:2])

        for iter_idx in range(niter):
            for time_idx in range(estimates_vol.shape[-1]):
                estimates_vol[:, :, :, time_idx] = (
                    (1 - mu) * estimates_vol[:, :, :, time_idx]
                    + mu * data_vol[:, :, :, time_idx]
                    - mu
                    * lambda_
                    * np.fft.ifftn(
                        h
                        * np.conj(h)
                        * np.fft.fftn(estimates_vol[:, :, :, time_idx], estimates_vol.shape[:2])
                    )
                )

    final_estimates = masker.fit_trasform(estimates_vol)

    return final_estimates


def spatial_structured_sparsity(estimates, data, mask, niter, dims, lambda_):
    r"""Spatial regularization technique based on the structured sparsity as in Total Activation.

    This function computes the structured sparsity regularization and is another variant of fgp
    algorithm for structured sparsity

    solves :math:`\\frac{1}{2} \\| \\mathbf{y} - \\mathbf{x} \\|^2 + \\lambda
    \\| \\mathbf{D}^\\textrm{order} \\cdot \\mathbf{x} \\|_{s,2,1}`.

    Delta is the laplacian operator delta[n] = [1 -2 1]; so symmetric, in
    matrix form :math:`\\Delta^T = \\Delta`.

    Parameters
    ----------
    estimates : ndarray
        Estimates (output of temporal regularization).
    data : ndarray
        Observations.
    mask : Nibabel object
        Mask image to unmask and mask estimates (2D to 4D and back).
    niter : int
        Number of iterations to perform spatial regularization.
    dims : list
        Dimensions of the data.
    lambda_ : float
        Spatial regularization parameter.

    Returns
    -------
    final_estimates: ndarray
        Estimates of activity-inducing or innovation signal after spatial regularization.
    """
    # Transform data from 2D into 4D
    estimates_vol = unmask(estimates, mask)
    data_vol = unmask(data, mask)

    z = np.zeros(estimates_vol.shape)

    h = generate_delta(dim=3)

    max_eig = 144

    h = np.fft.fftn(h, estimates_vol.shape[:2])

    # Perform structured sparsity regularization
    for time_idx in range(estimates_vol.shape[-1]):
        for iter_idx in range(niter):
            z[:, :, :, time_idx] = clip(
                z[:, :, :, time_idx]
                + 1
                / (lambda_ * max_eig)
                * np.fft.ifftn(h * np.fft.fftn(data_vol[:, :, :, time_idx], dims[:2]))
                - np.fft.ifftn(h * np.conj(h) * np.fft.fftn(z[:, :, :, time_idx], dims[:2]))
                / max_eig,
                mask,
            )
        estimates_vol[:, :, :, time_idx] = data_vol[:, :, :, time_idx] - lambda_ * np.fft.ifftn(
            np.conj(h) * np.fft.fttn(z[:, :, :, time_idx], dims[:2])
        )

    # Transform data from 4D into 2D
    final_estimates = apply_mask(estimates_vol, mask)

    return final_estimates


def clip(input, atlas):
    """Clip the input to the atlas.

    Parameters
    ----------
    input : ndarray
        Input to clip.
    atlas : ndarray
        Atlas to clip input to.

    Returns
    -------
    clipped_input: ndarray
        Clipped input.
    """
    clipped_input = np.zeros(input.shape)
    for region_idx in range(np.max(atlas)):
        # Find the indices of the voxels in the current region
        indices = np.where(atlas == region_idx + 1)

        if np.linalg.norm(input[indices]) > 1:
            # Clip the current region
            clipped_input[indices] = input[indices] / np.linalg.norm(input[indices])
        else:
            clipped_input[indices] = input[indices]

    return clipped_input


def generate_delta(dim=3):
    """Generate the delta operator.

    Parameters
    ----------
    dim : int, optional
        Number of dimensions of the operator, by default 3

    Returns
    -------
    h : ndarray
        The delta operator.

    Raises
    ------
    ValueError
        If dim is not 2 or 3.
    """
    if dim == 2:
        h = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    elif dim == 3:
        h = np.zeros((3, 3, 3))
        h[:, :, 0] = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        h[:, :, 1] = np.array([[0, 1, 0], [1, -6, 1], [0, 1, 0]])
        h[:, :, 2] = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    else:
        raise ValueError("Dimension must be 2 or 3")

    return h
