import numpy as np
from pywt import wavedec
from scipy.stats import median_absolute_deviation


def select_lambda(hrf, y, criteria="mad_update", factor=1, pcg=0.7, lambda_echo=-1):
    """Criteria to select regularization parameter lambda.

    Parameters
    ----------
    hrf : (E x T) array_like
        Matrix containing shifted HRFs in its columns. E stands for the number of volumes times
        the number of echo-times.
    y : (T x S) array_like
        Matrix with fMRI data provided to splora.
    criteria : str, optional
        Criteria to select regularization parameter lambda, by default "mad_update"
    factor : int, optional
        Factor by which to multiply the value of lambda, by default 1
        Only used when "factor" criteria is selected.
    pcg : str, optional
        Percentage of maximum lambda possible to use, by default "0.7"
        Only used when "pcg" criteria is selected.

    Returns
    -------
    lambda_selection : array_like
        Value of the regularization parameter lambda for each voxel.
    update_lambda : bool
        Whether to update lambda after each iteration until it converges to the MAD estimate
        of the noise.
    noise_estimate : array_like
        MAD estimate of the noise.
    """
    update_lambda = False
    nt = hrf.shape[1]

    # Use last echo to estimate noise
    if hrf.shape[0] > nt:
        if lambda_echo == -1:
            y = y[-nt:, :]
        else:
            y = y[(lambda_echo - 1) * nt : lambda_echo * nt, :]

    _, cD1 = wavedec(y, "db3", level=1, axis=0)
    noise_estimate = median_absolute_deviation(cD1) / 0.6745  # 0.8095

    if criteria == "mad":
        lambda_selection = noise_estimate
    elif criteria == "mad_update":
        lambda_selection = noise_estimate
        update_lambda = True
    elif criteria == "ut":
        lambda_selection = noise_estimate * np.sqrt(2 * np.log10(nt))
    elif criteria == "lut":
        lambda_selection = noise_estimate * np.sqrt(
            2 * np.log10(nt) - np.log10(1 + 4 * np.log10(nt))
        )
    elif criteria == "factor":
        lambda_selection = noise_estimate * factor
    elif criteria == "pcg":
        if pcg is None:
            raise ValueError("You must select a percentage to use the percentage criteria.")
        max_lambda = np.mean(abs(np.dot(hrf.T, y)), axis=0)
        lambda_selection = max_lambda * pcg
    elif criteria == "eigval":
        random_signal = np.random.normal(loc=0.0, scale=np.mean(noise_estimate), size=y.shape)
        s = np.linalg.svd(random_signal, compute_uv=False)
        lambda_selection = s[0]

    return lambda_selection, update_lambda, noise_estimate
