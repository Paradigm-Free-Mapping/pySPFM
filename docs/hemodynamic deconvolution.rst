#########################
Hemodynamic Deconvolution
#########################

Functional magnetic resonance imaging (fMRI) data analysis is often directed to identify and
disentangle the neural processes that occur in different brain regions during task or at rest.
As the blood oxygenation level-dependent (BOLD) signal of fMRI is only a proxy for neuronal
activity mediated through neurovascular coupling, an intermediate step that estimates the
activity-inducing signal, at the timescale of fMRI, from the BOLD timeseries can be useful.
Conventional analysis of task fMRI data relies on the general linear models (GLM) to establish
statistical parametric maps of brain activity by regression of the empirical timecourses against
hypothetical ones built from the knowledge of the experimental paradigm. However, timing
information of the paradigm can be unknown, inaccurate, or insufficient in some scenarios such as
naturalistic stimuli, resting-state, or clinically-relevant assessments.

Deconvolution methods estimate neuronal activity by undoing the blurring effect of the hemodynamic
response, characterized as a hemodynamic response function (HRF). Given
the inherently ill-posed nature of hemodynamic deconvolution, due to the strong temporal low-pass
characteristics of the HRF, the key is to introduce additional constraints in the estimation
problem that are typically expressed as regularizers.

If you would like to learn more about hemodynamic deconvolution, the following references may be
useful:

- Glover, G. H. (1999). `Deconvolution of impulse response in event-related BOLD fMRI`_. Neuroimage,
  9(4), 416-429.
- Gitelman, D. R., Penny, W. D., Ashburner, J., & Friston, K. J. (2003). `Modeling regional and
  psychophysiologic interactions in fMRI: the importance of hemodynamic deconvolution`_.
  Neuroimage, 19(1), 200-207.
- Gaudes, C. C., Petridou, N., Francis, S. T., Dryden, I. L., & Gowland, P. A. (2013).
  `Paradigm free mapping with sparse regression automatically detects single-trial functional
  magnetic resonance imaging blood oxygenation level dependent responses`_. Human brain mapping,
  34(3), 501.
- Karahanoğlu, F. I., Caballero-Gaudes, C., Lazeyras, F., & Van De Ville, D. (2013).
  `Total activation: fMRI deconvolution through spatio-temporal regularization`_. Neuroimage,
  73, 121-134.
- Uruñuela, E., Bolton, T. A., Van De Ville, D., & Caballero-Gaudes, C. (2021).
  `Hemodynamic Deconvolution Demystified: Sparsity-Driven Regularization at Work`_.
  arXiv preprint arXiv:2107.12026.

.. _Deconvolution of impulse response in event-related BOLD fMRI: https://doi.org/10.1006/nimg.1998.0419
.. _Modeling regional and psychophysiologic interactions in fMRI\: the importance of hemodynamic deconvolution: https://doi.org/10.1016/S1053-8119(03)00058-2
.. _Paradigm free mapping with sparse regression automatically detects single-trial functional magnetic resonance imaging blood oxygenation level dependent responses: https://doi.org/10.1002/hbm.21452
.. _Total activation\: fMRI deconvolution through spatio-temporal regularization: https://doi.org/10.1016/j.neuroimage.2013.01.067
.. _Hemodynamic Deconvolution Demystified\: Sparsity-Driven Regularization at Work: https://arxiv.org/abs/2107.12026