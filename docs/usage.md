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
timepoint, it tends to be detected across neighboring voxels simultaneously
([Uruñuela et al., 2024](https://doi.org/10.1016/j.media.2024.103010)):

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

For a comprehensive review of regularization parameter selection methods in hemodynamic
deconvolution, see [Uruñuela et al. (2023)](https://doi.org/10.52294/001c.87574).

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
  Robust noise estimation ([Karahanoğlu et al., 2013](https://doi.org/10.1016/j.neuroimage.2013.01.067); [Uruñuela et al., 2023](https://doi.org/10.52294/001c.87574)).

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

With HRF model selection:

```bash
# Using the Glover HRF
pySPFM sparse -i my_fmri_data.nii.gz -m my_mask.nii.gz \
    -o my_subject -d my_results_directory \
    -tr 2.0 --criterion bic --hrf-model glover

# Using a custom HRF from file
pySPFM sparse -i my_fmri_data.nii.gz -m my_mask.nii.gz \
    -o my_subject -d my_results_directory \
    -tr 2.0 --criterion mad --hrf-model /path/to/my_hrf.1D

# Using the block model for sustained activity
pySPFM sparse -i my_fmri_data.nii.gz -m my_mask.nii.gz \
    -o my_subject -d my_results_directory \
    -tr 2.0 --criterion factor --block
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
problem across multiple subsampled surrogate datasets
([Uruñuela et al., 2020](https://doi.org/10.1109/EMBC44109.2020.9175673)):

```bash
pySPFM stability -i my_fmri_data.nii.gz -m my_mask.nii.gz \
    -o my_subject_stability -d my_results_directory \
    -tr 2.0 --n-surrogates 50
```

### Low-Rank Plus Sparse (SPLORA)

For decomposing fMRI data into low-rank (structured noise) and sparse (neural activity) components
([Uruñuela et al., 2021](https://doi.org/10.1109/ISBI48211.2021.9434129)):

```bash
pySPFM lowrank -i my_fmri_data.nii.gz -m my_mask.nii.gz \
    -o my_subject_lowrank -d my_results_directory \
    -tr 2.0 --criterion factor
```

### AUC to Estimates

The `auc_to_estimates` workflow converts stability selection AUC (Area Under the Curve) maps
back into activity-inducing signal estimates. This is useful when you've run stability selection
and want to obtain debiased neural activity estimates from the AUC scores.

#### Python API

```python
from pySPFM.cli.auc_to_estimates import auc_to_estimates

# Basic usage: convert AUC to activity estimates
auc_to_estimates(
    data_fn=["my_fmri_data.nii.gz"],
    auc_fn="my_auc.nii.gz",
    mask_fn=["my_mask.nii.gz", "my_roi_mask.nii.gz"],
    output_filename="my_subject_estimates",
    tr=2.0,
    thr=95.0,  # 95th percentile threshold
    out_dir="my_results_directory",
)

# With time-dependent thresholding
auc_to_estimates(
    data_fn=["my_fmri_data.nii.gz"],
    auc_fn="my_auc.nii.gz",
    mask_fn=["my_mask.nii.gz", "my_roi_mask.nii.gz"],
    output_filename="my_subject_estimates",
    tr=2.0,
    thr=90.0,
    thr_strategy="time",  # Apply threshold at each TR
    out_dir="my_results_directory",
)

# Multi-echo data
auc_to_estimates(
    data_fn=["echo1.nii.gz", "echo2.nii.gz", "echo3.nii.gz"],
    auc_fn="my_auc.nii.gz",
    mask_fn=["my_mask.nii.gz"],
    output_filename="my_subject_estimates",
    tr=2.0,
    te=[14.5, 38.5, 62.5],  # Echo times in ms
    thr=95.0,
    out_dir="my_results_directory",
)

# Block model with grouping
auc_to_estimates(
    data_fn=["my_fmri_data.nii.gz"],
    auc_fn="my_auc.nii.gz",
    mask_fn=["my_mask.nii.gz", "my_roi_mask.nii.gz"],
    output_filename="my_subject_estimates",
    tr=2.0,
    thr=95.0,
    block_model=True,  # Estimate innovation signals
    group=True,        # Group consecutive coefficients
    group_distance=3,  # Max distance between grouped coefficients
    out_dir="my_results_directory",
)
```

#### Command Line Interface

```bash
auc_to_estimates -i my_fmri_data.nii.gz -a my_auc.nii.gz \
    -m my_mask.nii.gz my_auc_roi_mask.nii.gz \
    -o my_subject_estimates -d my_results_directory \
    -tr 2.0 -thr 95
```

**Key parameters:**

| Parameter | Description |
|-----------|-------------|
| `-i, --input` | Input fMRI data file(s) |
| `-a, --auc` | AUC map from stability selection |
| `-m, --mask` | Brain mask and optional thresholding mask |
| `-thr, --threshold` | Percentile threshold (1-100) or absolute threshold [0, 1) (i.e., 0 to less than 1). Default: 95 |
| `--strategy` | Thresholding strategy: `'static'` or `'time'` (time-dependent). Default: `'static'` |
| `-block, --block` | Estimate innovation signals (block model) |
| `--group` | Consider consecutive coefficients as belonging to the same block |
| `--group-distance` | Maximum distance between coefficients in the same block. Default: 3 |

**CLI example with time-dependent thresholding:**

```bash
auc_to_estimates -i my_fmri_data.nii.gz -a my_auc.nii.gz \
    -m my_mask.nii.gz my_roi_mask.nii.gz \
    -o my_subject_estimates -d my_results_directory \
    -tr 2.0 -thr 90 --strategy time
```

**CLI example with multi-echo data:**

```bash
auc_to_estimates -i echo1.nii.gz echo2.nii.gz echo3.nii.gz \
    -a my_auc.nii.gz -m my_mask.nii.gz \
    -o my_subject_estimates -d my_results_directory \
    -tr 2.0 -te 14.5 38.5 62.5 -thr 95
```

## HRF Model Configuration

The hemodynamic response function (HRF) is central to deconvolution. pySPFM supports
three ways to specify the HRF model via the `hrf_model` parameter:

### Built-in HRF Models

pySPFM provides two canonical HRF models from the literature:

```python
from pySPFM import SparseDeconvolution

# SPM canonical HRF (default)
model_spm = SparseDeconvolution(tr=2.0, hrf_model="spm")

# Glover HRF
model_glover = SparseDeconvolution(tr=2.0, hrf_model="glover")
```

| Model | Description |
|-------|-------------|
| `'spm'` | SPM canonical HRF with default parameters (default) |
| `'glover'` | Glover HRF model from FSL/FreeSurfer |

Both models are implemented via [nilearn](https://nilearn.github.io/) and are commonly
used in fMRI analysis. The SPM model is the default as it is widely adopted.

### Custom HRF from File

For advanced users who need a specific HRF shape (e.g., from a separate HRF estimation
procedure), you can provide a custom HRF as a `.1D` or `.txt` file:

```python
from pySPFM import SparseDeconvolution

# Use a custom HRF from a text file
model = SparseDeconvolution(
    tr=2.0,
    hrf_model="/path/to/my_custom_hrf.1D",
)
```

**Custom HRF file requirements:**

- Format: Plain text file with one value per line (`.1D` or `.txt` extension)
- Length: Must not exceed the number of scans in your data
- Sampling: Should be sampled at the TR of your acquisition

Example custom HRF file (`my_hrf.1D`):
```
0.0
0.1
0.5
1.0
0.8
0.4
0.1
0.0
-0.1
-0.05
0.0
```

### Block vs Spike Model

The `block_model` parameter controls what type of signal is estimated:

```python
# Spike model (default): estimate activity-inducing signals
model_spike = SparseDeconvolution(
    tr=2.0,
    block_model=False,  # Default
)

# Block model: estimate innovation signals (step functions)
model_block = SparseDeconvolution(
    tr=2.0,
    block_model=True,
)
```

| Parameter | Signal Type | Description |
|-----------|-------------|-------------|
| `block_model=False` | Activity-inducing | Neural events as impulses (spikes) |
| `block_model=True` | Innovation | Sustained activity as step functions (blocks) |

When `block_model=True`, the HRF matrix is modified to include an integrator
(cumulative sum), which models sustained activity that starts at one timepoint
and continues. This is useful for paradigm-free mapping of block-like neural responses.

### Accessing the HRF Matrix

After fitting, you can inspect the HRF convolution matrix:

```python
model = SparseDeconvolution(tr=2.0, hrf_model="glover")
model.fit(X)

# The HRF matrix used for deconvolution
print(f"HRF matrix shape: {model.hrf_matrix_.shape}")
# Shape: (n_timepoints * n_echoes, n_timepoints)
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

### Using different HRF models

```python
from pySPFM import SparseDeconvolution
import numpy as np

X = np.random.randn(200, 100)

# Using the Glover HRF instead of SPM
model_glover = SparseDeconvolution(
    tr=2.0,
    hrf_model="glover",
    criterion="bic",
)
model_glover.fit(X)

# Using a custom HRF from file
model_custom = SparseDeconvolution(
    tr=2.0,
    hrf_model="/path/to/my_estimated_hrf.1D",
    criterion="mad",
)
# model_custom.fit(X)  # Uncomment with valid path
```

### Block model for sustained activity

```python
from pySPFM import SparseDeconvolution
import numpy as np

X = np.random.randn(200, 100)

# Use block model for paradigm-free mapping of sustained activity
model = SparseDeconvolution(
    tr=2.0,
    block_model=True,  # Estimate innovation signals (step functions)
    criterion="factor",
    factor=1.0,
)
model.fit(X)

# coef_ now contains innovation signals representing
# onsets/offsets of sustained activity
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

### Core Methodology

- [Uruñuela et al. (2023)](https://doi.org/10.52294/001c.87574):
  **Hemodynamic Deconvolution Demystified: Sparsity-Driven Regularization at Work.**
  *Aperture Neuro.* Comprehensive review of sparse deconvolution methods and regularization strategies.

- [Caballero-Gaudes et al. (2013)](https://doi.org/10.1002/hbm.21452):
  Paradigm free mapping with sparse regression. *Human Brain Mapping.*

- [Karahanoğlu et al. (2013)](https://doi.org/10.1016/j.neuroimage.2013.01.067):
  Total activation: fMRI deconvolution through spatio-temporal regularization. *NeuroImage.*

### Stability Selection

- [Uruñuela et al. (2020)](https://doi.org/10.1109/EMBC44109.2020.9175673):
  **Stability-based sparse paradigm free mapping algorithm for deconvolution of functional MRI data.**
  *IEEE EMBC.* Introduces the stability selection approach for robust sparse deconvolution.

### Multivariate / Whole-Brain Deconvolution

- [Uruñuela et al. (2024)](https://doi.org/10.1016/j.media.2024.103010):
  **Whole-brain multivariate hemodynamic deconvolution for functional MRI with stability selection.**
  *Medical Image Analysis.* Extends sparse deconvolution to whole-brain multivariate analysis.

- [Uruñuela et al. (2019)](https://cds.ismrm.org/protected/19MProceedings/PDFfiles/3371.html):
  Deconvolution of multi-echo functional MRI data with multivariate multi-echo sparse paradigm free mapping.
  *ISMRM.*

### Low-Rank Plus Sparse (SPLORA)

- [Uruñuela et al. (2021)](https://doi.org/10.1109/ISBI48211.2021.9434129):
  **A low rank and sparse paradigm free mapping algorithm for deconvolution of fMRI data.**
  *IEEE ISBI.* Introduces the SPLORA algorithm for separating global and neural components.
