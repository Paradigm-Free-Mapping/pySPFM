# Usage

`pySPFM` provides scikit-learn compatible estimators for sparse hemodynamic deconvolution of fMRI data.
There are two main approaches: using a fixed regularization parameter $\lambda$, or using stability
selection for more robust but computationally intensive estimation.

## Python API

The recommended way to use pySPFM is through the Python API with scikit-learn style estimators:

```python
from pySPFM import SparseDeconvolution
import nibabel as nib
from nilearn.maskers import NiftiMasker

# Load and mask your fMRI data
img = nib.load("my_fmri_data.nii.gz")
masker = NiftiMasker(mask_img="my_mask.nii.gz")
X = masker.fit_transform(img)  # Shape: (n_timepoints, n_voxels)

# Fit the deconvolution model
model = SparseDeconvolution(tr=2.0, criterion="bic")
model.fit(X)

# Get deconvolved activity-inducing signals
activity = model.coef_  # Shape: (n_timepoints, n_voxels)
```

## Univariate vs Multivariate Mode

The `SparseDeconvolution` estimator supports two solving modes controlled by the `group` parameter.
This is a key design decision that affects how the regularization problem is solved.

### Univariate Mode (default)

When `group=0.0` (the default), each voxel is solved **independently** using pure L1 (LASSO)
regularization. This is the standard approach and is computationally efficient:

```python
# Univariate mode: each voxel solved independently
model = SparseDeconvolution(
    tr=2.0,
    criterion="bic",  # Can use LARS criteria (bic, aic)
    group=0.0,        # Pure L1/LASSO regularization
    n_jobs=4,         # Parallelize across voxels
)
model.fit(X)
```

**In univariate mode:**

- Each voxel has its **own** regularization parameter $\lambda$
- Computation can be **parallelized** across voxels via `n_jobs`
- **All criteria** are available: `'bic'`, `'aic'`, `'mad'`, `'factor'`, etc.
- Best for exploratory analysis or when spatial structure is not important

### Multivariate Mode (Spatial Grouping)

When `group > 0`, all voxels are solved **jointly** using L2,1 mixed-norm regularization.
This encourages **spatial grouping** of activity—if a neural event occurs at a particular
timepoint, it tends to be detected across neighboring voxels simultaneously:

```python
# Multivariate mode: all voxels solved jointly with spatial regularization
model = SparseDeconvolution(
    tr=2.0,
    criterion="factor",  # Must use FISTA criteria (not bic/aic)
    group=0.5,           # L2,1 + L1 mixed-norm regularization
)
model.fit(X)
```

**The `group` parameter** controls the balance between L1 and L2,1 norms:

| Value | Regularization | Description |
|-------|---------------|-------------|
| `group=0.0` | Pure L1/LASSO | Univariate, no spatial structure |
| `0 < group < 1` | L1 + L2,1 mix | Elastic net-like combination |
| `group=1.0` | Pure L2,1 | Maximum spatial grouping |

**In multivariate mode:**

- A **single** $\lambda$ is used for all voxels
- Computation **cannot** be parallelized (joint optimization)
- Only **FISTA criteria** are available: `'mad'`, `'factor'`, `'ut'`, `'lut'`, `'pcg'`, `'eigval'`
- Leverages the spatial structure of 4D fMRI data

**When to use multivariate mode:**

- When you expect spatially coherent neural activity
- When analyzing regions of interest (ROIs) with known spatial structure
- When you want to leverage the 4D structure of fMRI data
- When reducing false positives through spatial consistency is important

### Mathematical Background

The deconvolution problem is formulated as:

$$\min_{\mathbf{s}} \frac{1}{2} \|\mathbf{y} - \mathbf{H}\mathbf{s}\|_2^2 + \lambda \cdot R(\mathbf{s})$$

where:
- $\mathbf{y}$ is the BOLD signal
- $\mathbf{H}$ is the HRF convolution matrix
- $\mathbf{s}$ is the activity-inducing signal
- $R(\mathbf{s})$ is the regularization term

**Univariate mode** (`group=0`): $R(\mathbf{s}) = \|\mathbf{s}\|_1$ (L1 norm, LASSO)

**Multivariate mode** (`group > 0`):
$R(\mathbf{s}) = (1 - \text{group}) \|\mathbf{s}\|_1 + \text{group} \|\mathbf{S}\|_{2,1}$

where $\|\mathbf{S}\|_{2,1} = \sum_t \|\mathbf{s}_t\|_2$ is the L2,1 mixed norm that encourages
rows of the coefficient matrix (timepoints) to be jointly zero or non-zero across voxels.

## Lambda Selection Criteria

### LARS-based criteria (univariate only)

The following criteria use the Least Angle Regression (LARS) algorithm to efficiently
compute the full regularization path and select $\lambda$ using information criteria:

- `'bic'`: Bayesian Information Criterion - penalizes model complexity more heavily
- `'aic'`: Akaike Information Criterion - less penalty for model complexity

```{note}
LARS criteria (`'bic'`, `'aic'`) are only available in univariate mode (`group=0`).
They cannot be used with spatial regularization because LARS solves each voxel independently.
```

### FISTA-based criteria (univariate and multivariate)

The following criteria use the FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)
solver and support both univariate and multivariate modes:

- `'mad'`: Median Absolute Deviation of fine-scale wavelet coefficients (Daubechies-3).
  Robust noise estimation as in [Karahanoğlu et al. (2013)].

- `'mad_update'`: MAD with iterative updating. The noise estimate is refined during optimization:
  $\lambda_{n+1} = \frac{N \sigma}{\frac{1}{2} \|\mathbf{y} - \mathbf{Hs}\|_2^2 \lambda_n}$

- `'ut'`: Universal Threshold:
  $\lambda = \sigma \sqrt{2 \log(T)}$, where $T$ is the number of timepoints.

- `'lut'`: Lower Universal Threshold:
  $\lambda = \sigma \sqrt{2 \log(T) - \log(1 + 4\log(T))}$

- `'factor'`: Factor of noise estimate:
  $\lambda = \text{factor} \times \sigma$, controlled by the `factor` parameter.

- `'pcg'`: Percentage of maximum lambda:
  $\lambda = \text{pcg} \times \|\mathbf{H}^T \mathbf{y}\|_\infty$, controlled by the `pcg` parameter.

- `'eigval'`: Based on eigenvalue decomposition of the HRF matrix.

## Command Line Interface

pySPFM also provides a command-line interface for common workflows.

### Sparse Deconvolution

```bash
pySPFM sparse -i my_fmri_data.nii.gz -m my_mask.nii.gz \
    -o my_subject -d my_results_directory \
    -tr 2.0 --criterion mad
```

For multi-echo data:

```bash
pySPFM sparse -i echo1.nii.gz echo2.nii.gz echo3.nii.gz \
    -m my_mask.nii.gz -te 14.5 38.5 62.5 \
    -o my_subject -d my_results_directory \
    -tr 2.0 --criterion mad
```

### Stability Selection

The stability selection procedure provides more robust estimates by solving the regularization
problem across multiple subsampled surrogate datasets:

```bash
pySPFM stability -i my_fmri_data.nii.gz -m my_mask.nii.gz \
    -o my_subject_stability -d my_results_directory \
    -tr 2.0 --n-surrogates 50
```

### Low-Rank Plus Sparse

For decomposing fMRI data into low-rank (structured noise) and sparse (neural activity) components:

```bash
pySPFM lowrank -i my_fmri_data.nii.gz -m my_mask.nii.gz \
    -o my_subject_lowrank -d my_results_directory \
    -tr 2.0 --criterion factor
```

## Examples

### Basic single-echo deconvolution

```python
from pySPFM import SparseDeconvolution
import numpy as np

# Simulate data: 200 timepoints, 100 voxels
X = np.random.randn(200, 100)

# Univariate deconvolution with BIC
model = SparseDeconvolution(tr=2.0, criterion="bic")
model.fit(X)

print(f"Coefficients shape: {model.coef_.shape}")
print(f"Lambda values shape: {model.lambda_.shape}")
```

### Multi-echo deconvolution

```python
from pySPFM import SparseDeconvolution
import numpy as np

# Multi-echo data: stack echoes along time axis
# 3 echoes × 200 timepoints = 600 rows
X = np.random.randn(600, 100)

model = SparseDeconvolution(
    tr=2.0,
    te=[14.5, 38.5, 62.5],  # Echo times in ms
    criterion="mad",
)
model.fit(X)
```

### Multivariate deconvolution with spatial grouping

```python
from pySPFM import SparseDeconvolution
import numpy as np

X = np.random.randn(200, 100)

# Use spatial grouping to encourage consistent activity across voxels
model = SparseDeconvolution(
    tr=2.0,
    criterion="factor",  # FISTA criterion required for group > 0
    group=0.5,           # 50% L2,1 + 50% L1
    factor=1.0,
)
model.fit(X)

# In multivariate mode, a single lambda is used for all voxels
print(f"Lambda (same for all voxels): {model.lambda_[0]}")
```

### Stability selection for robust estimation

```python
from pySPFM import StabilitySelection
import numpy as np

X = np.random.randn(200, 100)

model = StabilitySelection(
    tr=2.0,
    n_surrogates=50,      # Number of subsampled datasets
    subsample_fraction=0.5,
)
model.fit(X)

# Get the stability scores (AUC)
auc = model.auc_
```

## References

- [Karahanoğlu et al. (2013)](https://doi.org/10.1016/j.neuroimage.2013.01.067):
  Total activation: fMRI deconvolution through spatio-temporal regularization.
- [Caballero-Gaudes et al. (2013)](https://doi.org/10.1002/hbm.21452):
  Paradigm free mapping with sparse regression.
- [Uruñuela et al. (2023)](https://doi.org/10.52294/001c.87574):
  Hemodynamic Deconvolution Demystified: Sparsity-Driven Regularization at Work.
