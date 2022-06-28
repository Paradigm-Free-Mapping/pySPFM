"""Selection of the regularization parameter lambda for the deconvolution."""
import numpy as np
from pywt import wavedec
from scipy.stats import median_absolute_deviation


def select_lambda(hrf, y, criterion="ut", factor=1, pcg=0.7, lambda_echo=-1):
    """
    Criteria to select regularization parameter lambda.

    Parameters
    ----------
    hrf : (E x T) array_like
        Matrix containing shifted HRFs in its columns. E stands for the number of volumes times
        the number of echo-times.
    y : (T x S) array_like
        Matrix with fMRI data provided to pySPFM.
    criterion : str, optional
        Criteria to select regularization parameter lambda, by default "ut".
    factor : int, optional
        Factor by which to multiply the value of lambda, by default 1
        Only used when 'factor' criterion is selected.
    pcg : str, optional
        Percentage of maximum lambda possible to use, by default '0.7'
        Only used when 'pcg' criterion is selected.

    Returns
    -------
    lambda_selection : array_like
        Value of the regularization parameter lambda for each voxel.
    update_lambda : bool
        Whether to update lambda after each iteration until it converges to the MAD estimate
        of the noise.
    noise_estimate : array_like
        MAD estimate of the noise.

    Notes
    -----
    The criteria to select the regularization parameter lambda are:

    - 'ut': universal threshold.
        :math:`{\\lambda} = {\\sigma} * \\sqrt{2 * \\log(T)}`, where :math:`{\\sigma}` is the
        median absolute deviation of the estimated level of noise and T is the number of TRs.
    - 'lut' : lower universal threshold.
        :math:`\\lambda = \\sigma * \\sqrt{2 * \\log(T) - \\log(1 + 4 * \\log(T))}`, where
        :math:`\\sigma` is the median absolute deviation of the estimated level of noise and T is
        the number of TRs.
    - 'mad' : mediam absolute deviation.
        Calculate lambda as the median absolute deviation of fine-scale wavelet
        coefficients (Daubechies, order 3). For more information,
        see Karahanoglu et al. (2013).
    - 'mad_update' : updating median absolute deviation.
        Median absolute deviation of the estimated level of the noise that gets
        updated on each iteration (see Karahanoglu et al. 2013):
        :math:`\\lambda_{n+1} = {\\frac{N \\sigma}{1/2 \|
        \\mathbf{y} - \\mathbf{x} \|_2^2 \\lambda_n}}`.
    - 'pcg' : percentage of the maximum lambda possible to use as lambda.
        :math:`\\lambda = \\textrm{pcg} * \\lambda_{max}`,
        where :math:`\\lambda_{max}= \| \\mathbf{H}^T \\mathbf{y} \|` and
        :math:`0 \\leq \\textrm{pcg} \\leq 1`
    - 'factor' : factor of the estimate of the level of noise to use as lambda.
        :math:`\\lambda = \\textrm{factor} * \\sigma, with 0 \\leq \\textrm{factor} \\leq 1`
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

    if criterion == "mad":
        lambda_selection = noise_estimate
    elif criterion == "mad_update":
        lambda_selection = noise_estimate
        update_lambda = True
    elif criterion == "ut":
        lambda_selection = noise_estimate * np.sqrt(2 * np.log10(nt))
    elif criterion == "lut":
        lambda_selection = noise_estimate * np.sqrt(
            2 * np.log10(nt) - np.log10(1 + 4 * np.log10(nt))
        )
    elif criterion == "factor":
        lambda_selection = noise_estimate * factor
    elif criterion == "pcg":
        if pcg is None:
            raise ValueError("You must select a percentage to use the percentage criterion.")
        max_lambda = np.mean(abs(np.dot(hrf.T, y)), axis=0)
        lambda_selection = max_lambda * pcg
    elif criterion == "eigval":
        random_signal = np.random.normal(loc=0.0, scale=np.mean(noise_estimate), size=y.shape)
        s = np.linalg.svd(random_signal, compute_uv=False)
        lambda_selection = s[0]

    return lambda_selection, update_lambda, noise_estimate
