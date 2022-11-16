#####
Usage
#####

`pySPFM` has two main ways of usage: using a fixed selection of the regularization parameter lambda,
or using the stability selection method, which avoids the selection of lambda and yields more robust
estimates of the neuronal-related signal at the cost of higher computation time.

*********************
Fixed :math:`\lambda`
*********************

The fixed :math:`\lambda` method is the simplest one. It consists in choosing one of the following methods
to automatically calculate the regularization parameter lambda:

- `universal threshold`:
    :math:`\lambda = \sigma * \sqrt{2 * \log(T)}`, where :math:`\sigma` is the median absolute
    deviation of the estimated level of noise and T is the number of TRs.
- `lower universal threshold`:
    :math:`\lambda = \sigma * \sqrt{2 * \log(T) - \log(1 + 4 * \log(T))}`, where :math:`\sigma`
    is the median absolute deviation of the estimated level of noise and T is the number of TRs.
- `median absolute deviation`:
    Calculate lambda as the median absolute deviation of fine-scale wavelet coefficients
    (Daubechies, order 3). For more information, see `Karahanoglu et al. (2013)`_.
- `updating median absolute deviation`:
    Median absolute deviation of the estimated level of the noise that gets updated on each
    iteration (see `Karahanoglu et al. (2013)`_):
    :math:`\lambda_{n+1} = {\frac{N \sigma}{\frac{1}{2} \| \mathbf{y} - \mathbf{Hs} \|_2^2 \lambda_n}}`.
- `percentage of maximum lambda`:
    percentage of the maximum lambda possible to use as lambda.
    :math:`\lambda = \textrm{pcg} * \lambda_{max}`,
    where :math:`\lambda_{max}= \| \mathbf{H}^T \mathbf{y} \|` and
    :math:`0 \leq \textrm{pcg} \leq 1`
- `factor of median absolute deviation`:
    factor of the estimate of the level of noise to use as lambda.
    :math:`\lambda = \textrm{factor} * \sigma, with 0 \leq \textrm{factor} \leq 1`

.. code-block:: bash

    pySPFM -i my_echo_1.nii.gz my_echo_3.nii.gz my_echo_3.nii.gz -m my_mask.nii.gz
    -te 14.5, 38.5, 62.5 -o my_subject_pySPFM -tr 2 -d my_results_directory -crit mad

.. _Karahanoglu et al. (2013): https://10.1016/j.neuroimage.2013.01.067

*******************
Stability selection
*******************

The stability selection procedure solves the regularization problem in a number of subsampled
surrogate datasets with the Least Angle Regression algorithm. 

.. code-block:: bash

    pySPFM -i my_data.nii.gz -m my_mask.nii.gz -o my_subject_stability_selection -tr 2
    -d my_results_directory -crit stability -nsur 50