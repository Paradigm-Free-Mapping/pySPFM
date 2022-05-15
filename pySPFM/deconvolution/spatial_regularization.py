import numpy as np
from nilearn.masking import apply_mask, unmask


def spatial_tikhonov(estimates, data, mask, niter, dim, lambda_, mu):

    # Transform data from 2D into 4D
    estimates_vol = unmask(estimates, mask)
    data_vol = unmask(data, mask)

    if dim == 2:
        h = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

        # TODO: find fft2 equivalent function for Python
        H = fft2(h, estimates_vol.shape[0], estimates_vol.shape[1])

        for iter_idx in range(niter):
            for time_idx in range(estimates_vol.shape[-1]):
                for slice_idx in range(estimates_vol.shape[2]):
                    # TODO: find ifft2 and conj equivalent for Python
                    estimates_vol[:, :, slice_idx, time_idx] = (
                        (1 - mu) * estimates_vol[:, :, slice_idx, time_idx]
                        + mu * data_vol[:, :, slice_idx, time_idx]
                        - mu
                        * lambda_
                        * ifft2(
                            (conj(H) * H)
                            * fft2(
                                estimates_vol[:, :, slice_idx, time_idx],
                                estimates_vol.shape[0],
                                estimates_vol.shape[1],
                            )
                        )
                    )

    elif dim == 3:
        h = np.zeros((3, 3, 3))
        h[:, :, 0] = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        h[:, :, 1] = np.array([[0, 1, 0], [1, -6, 1], [0, 1, 0]])
        h[:, :, 2] = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

        # TODO: find fftn equivalent for Python
        H = fftn(h, estimates_vol.shape[:2])

        for iter_idx in range(niter):
            for time_idx in range(estimates_vol.shape[-1]):
                estimates_vol[:, :, :, time_idx] = (
                    (1 - mu) * estimates_vol[:, :, :, time_idx]
                    + mu * data_vol[:, :, :, time_idx]
                    - mu
                    * lambda_
                    * ifftn(
                        H
                        * conj(H)
                        * fftn(estimates_vol[:, :, :, time_idx], estimates_vol.shape[:2])
                    )
                )

    final_estimates = apply_mask(estimates_vol, mask)

    return final_estimates
