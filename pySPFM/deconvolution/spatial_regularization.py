import numpy as np
from nilearn.masking import apply_mask, unmask


def spatial_tikhonov(estimates, data, mask, niter, dim, lambda_, mu):
    """Spatial regularization technique based on the Tikhonov regularization as in Total Activation.

    This function computes the tikhonov regularization

    F(x) = min ||y - x ||^2 + lambda * ||Delta{x}||^2

    Delta is the laplacian operator delta[n] = [1 -2 1]; so symmetric, in
    matrix form Delta^T = Delta.

    Parameters
    ----------
    estimates : numpy.array
        Estimates (output of temporal regularization).
    data : numpy.array
        Observations.
    mask : Nibabel object
        Mask image to unmask and mask estimates (2D to 4D and back).
    niter : int
        Number of iterations to perform spatial regularization.
    dim : int
        Slice-wise regularization with dim = 2; whole-volume regularization with dim=3. Default = 3.
    lambda_ : float
        Spatial regularization parameter. Default = 1.
    mu : float
        Step size (small, ~0.01)

    Returns
    -------
    final_estimates: np.array
        Estimates of activity-inducing or innovation signal after spatial regularization.
    """
    # Transform data from 2D into 4D
    estimates_vol = unmask(estimates, mask)
    data_vol = unmask(data, mask)

    if dim == 2:
        h = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

        H = np.fft.fft2(h, (estimates_vol.shape[0], estimates_vol.shape[1]))

        for iter_idx in range(niter):
            for time_idx in range(estimates_vol.shape[-1]):
                for slice_idx in range(estimates_vol.shape[2]):
                    estimates_vol[:, :, slice_idx, time_idx] = (
                        (1 - mu) * estimates_vol[:, :, slice_idx, time_idx]
                        + mu * data_vol[:, :, slice_idx, time_idx]
                        - mu
                        * lambda_
                        * np.fft.ifft2(
                            (np.conj(H) * H)
                            * np.fft.fft2(
                                estimates_vol[:, :, slice_idx, time_idx],
                                (estimates_vol.shape[0], estimates_vol.shape[1]),
                            )
                        )
                    )

    elif dim == 3:
        h = np.zeros((3, 3, 3))
        h[:, :, 0] = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        h[:, :, 1] = np.array([[0, 1, 0], [1, -6, 1], [0, 1, 0]])
        h[:, :, 2] = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

        H = np.fft.fftn(h, estimates_vol.shape[:2])

        for iter_idx in range(niter):
            for time_idx in range(estimates_vol.shape[-1]):
                estimates_vol[:, :, :, time_idx] = (
                    (1 - mu) * estimates_vol[:, :, :, time_idx]
                    + mu * data_vol[:, :, :, time_idx]
                    - mu
                    * lambda_
                    * np.fft.ifftn(
                        H
                        * np.conj(H)
                        * np.fft.fftn(estimates_vol[:, :, :, time_idx], estimates_vol.shape[:2])
                    )
                )

    final_estimates = apply_mask(estimates_vol, mask)

    return final_estimates
