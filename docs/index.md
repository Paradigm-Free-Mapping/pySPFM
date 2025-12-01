```{eval-rst}
.. include:: <isonum.txt>
```

# pySPFM

**Sparse Paradigm Free Mapping for fMRI**

```{image} https://img.shields.io/pypi/v/pySPFM.svg
:alt: Latest Version
:target: https://pypi.python.org/pypi/pySPFM/
```

```{image} https://img.shields.io/pypi/pyversions/pySPFM.svg
:alt: PyPI - Python Version
:target: https://pypi.python.org/pypi/pySPFM/
```

```{image} https://zenodo.org/badge/492450151.svg
:alt: DOI
:target: https://zenodo.org/badge/latestdoi/492450151
```

```{image} https://circleci.com/gh/Paradigm-Free-Mapping/pySPFM/tree/main.svg?style=shield
:alt: CircleCI
:target: https://circleci.com/gh/Paradigm-Free-Mapping/pySPFM/tree/main
```

```{image} http://img.shields.io/badge/License-LGPL%202.1-blue.svg
:alt: License
:target: https://opensource.org/licenses/LGPL-2.1
```

```{image} https://readthedocs.org/projects/pyspfm/badge/?version=latest
:alt: Documentation Status
:target: http://pyspfm.readthedocs.io/en/latest/?badge=latest
```

```{image} https://codecov.io/gh/Paradigm-Free-Mapping/pySPFM/branch/main/graph/badge.svg
:alt: Codecov
:target: https://codecov.io/gh/Paradigm-Free-Mapping/pySPFM
```

## About

`pySPFM` is a Python library for **sparse hemodynamic deconvolution** of fMRI data.
It provides scikit-learn compatible estimators for detecting neural activity without
requiring prior knowledge of experimental timing—making it ideal for naturalistic,
resting-state, and clinical fMRI studies.

### Why pySPFM?

Traditional fMRI analysis relies on knowing *when* stimuli were presented. But what if you:

- 🧠 **Don't have timing information** — analyzing resting-state or naturalistic paradigms
- 🏥 **Study spontaneous events** — epileptic spikes, mind wandering, or intrinsic activity
- 🔬 **Want data-driven discovery** — letting the data reveal *when* neural events occurred
- 📊 **Need robust estimation** — reducing false positives through stability selection

**pySPFM solves these problems** by deconvolving the hemodynamic response function (HRF)
from BOLD signals to estimate the underlying neural activity at each timepoint.

## Key Features

### 🎯 Sparse Deconvolution (`SparseDeconvolution`)

The core algorithm for paradigm-free mapping. Estimates sparse activity-inducing signals
using L1 (LASSO) or L2,1 mixed-norm regularization.

- **Univariate mode**: Fast, voxel-wise independent deconvolution
- **Multivariate mode**: Joint spatial regularization for whole-brain consistency
- **Multiple solvers**: LARS (fast, exact) or FISTA (flexible, supports spatial grouping)
- **Automatic λ selection**: BIC, AIC, MAD, and other criteria

### 🔄 Low-Rank Plus Sparse (`LowRankPlusSparse`)

The SPLORA algorithm separates fMRI data into:

- **Low-rank component**: Global/structured signals (physiological noise, drift)
- **Sparse component**: Transient neuronal activity

Perfect for cleaning data while preserving neural events.

### 📊 Stability Selection (`StabilitySelection`)

Robust feature selection through subsampling. Instead of a single solution, get
**selection frequencies** that indicate how reliably each timepoint is detected
across many subsampled datasets.

### 🔧 Flexible & Extensible

- **scikit-learn API**: `fit()`, `transform()`, `get_params()`, `set_params()`
- **Multi-echo support**: Leverage information across echo times
- **Custom HRF**: Use SPM, Glover, or your own HRF model
- **Block/Spike models**: Detect transient events or sustained activity

## Quick Start

```python
from pySPFM import SparseDeconvolution
import nibabel as nib
from nilearn.maskers import NiftiMasker

# Load and mask fMRI data
img = nib.load("my_fmri_data.nii.gz")
masker = NiftiMasker(mask_img="my_mask.nii.gz")
X = masker.fit_transform(img)  # (n_timepoints, n_voxels)

# Fit sparse deconvolution
model = SparseDeconvolution(tr=2.0, criterion="bic")
model.fit(X)

# Get estimated neural activity
activity = model.coef_  # (n_timepoints, n_voxels)
```

## References

When using pySPFM, please cite the following:

### Core Methodology

1. **Uruñuela, E.**, Bolton, T. A. W., Van De Ville, D., & Caballero-Gaudes, C. (2023).
   [Hemodynamic Deconvolution Demystified: Sparsity-Driven Regularization at Work.](https://doi.org/10.52294/001c.87574)
   *Aperture Neuro*, 3, 1–25.

2. Caballero-Gaudes, C., Petridou, N., Francis, S. T., Dryden, I. L., & Gowland, P. A. (2013).
   [Paradigm free mapping with sparse regression automatically detects single-trial fMRI responses.](https://doi.org/10.1002/hbm.21452)
   *Human Brain Mapping*, 34(3), 501–518.

3. Karahanoğlu, F. I., Caballero-Gaudes, C., Lazeyras, F., & Van De Ville, D. (2013).
   [Total activation: fMRI deconvolution through spatio-temporal regularization.](https://doi.org/10.1016/j.neuroimage.2013.01.067)
   *NeuroImage*, 73, 121–134.

### Stability Selection

4. **Uruñuela, E.**, Jones, S., Crawford, A., et al. (2020).
   [Stability-based sparse paradigm free mapping algorithm for deconvolution of functional MRI data.](https://doi.org/10.1109/EMBC44109.2020.9175673)
   *IEEE EMBC*, 1092–1095.

### Multivariate / Whole-Brain Deconvolution

5. **Uruñuela, E.**, Gonzalez-Castillo, J., Zheng, C., Bandettini, P. A., & Caballero-Gaudes, C. (2024).
   [Whole-brain multivariate hemodynamic deconvolution for functional MRI with stability selection.](https://doi.org/10.1016/j.media.2024.103010)
   *Medical Image Analysis*, 91, 103010.

### Low-Rank Plus Sparse (SPLORA)

6. **Uruñuela, E.**, Moia, S., & Caballero-Gaudes, C. (2021).
   [A low rank and sparse paradigm free mapping algorithm for deconvolution of fMRI data.](https://doi.org/10.1109/ISBI48211.2021.9434129)
   *IEEE ISBI*, 1726–1729.

### Software Citation

> ```{raw} html
> <script language="javascript">
> var version = 'latest';
> function fillCitation(){
>    $('#pySPFM_version').text(version);
>
>    function cb(err, zenodoID) {
>       getCitation(zenodoID, 'vancouver-brackets-no-et-al', function(err, citation) {
>          $('#pySPFM_citation').text(citation);
>       });
>       getDOI(zenodoID, function(err, DOI) {
>          $('#pySPFM_doi_url').text('https://doi.org/' + DOI);
>          $('#pySPFM_doi_url').attr('href', 'https://doi.org/' + DOI);
>       });
>    }
>
>    if(version == 'latest') {
>       getLatestIDFromconceptID("6600095", cb);
>    } else {
>       getZenodoIDFromTag("6600095", version, cb);
>    }
> }
> </script>
> <p>
> <span id="pySPFM_citation">pySPFM</span> —
> <a id="pySPFM_doi_url" href="https://doi.org/10.5281/zenodo.6600095">https://doi.org/10.5281/zenodo.6600095</a>
> <img src onerror='fillCitation()' alt=""/>
> </p>
> ```

## License

pySPFM is licensed under the [GNU Lesser General Public License version 2.1](https://opensource.org/licenses/LGPL-2.1).

```{toctree}
:caption: 'Contents:'
:maxdepth: 2

installation
hemodynamic deconvolution
usage
api
examples
outputs
```

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
