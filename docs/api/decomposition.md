
# Decomposition (Estimators)

The main scikit-learn compatible estimators for sparse hemodynamic deconvolution.

```{eval-rst}
.. currentmodule:: pySPFM

.. autosummary::
   :toctree: ../generated/
   :template: class.rst

   SparseDeconvolution
   LowRankPlusSparse
   StabilitySelection
```

## Overview

pySPFM provides three main estimators that follow the scikit-learn API:

- **{class}`SparseDeconvolution`**: The core estimator for sparse hemodynamic deconvolution.
  Supports both univariate (voxel-wise) and multivariate (joint spatial) solving modes.

- **{class}`LowRankPlusSparse`**: Decomposes fMRI data into low-rank (structured noise/drift)
  and sparse (neural activity) components using robust PCA combined with sparse deconvolution.

- **{class}`StabilitySelection`**: Provides robust estimation through stability selection,
  solving the problem across multiple subsampled datasets and computing area-under-curve (AUC)
  stability scores.

## Univariate vs Multivariate Mode

The `SparseDeconvolution` estimator's `group` parameter controls whether voxels are solved
independently or jointly:

| Mode | `group` value | Regularization | Parallelization | Criteria |
|------|--------------|----------------|-----------------|----------|
| **Univariate** | `0.0` | L1 (LASSO) | ✅ via `n_jobs` | All |
| **Multivariate** | `> 0` | L1 + L2,1 | ❌ Joint solve | FISTA only |

See the {doc}`../usage` page for detailed examples and mathematical background.

## SparseDeconvolution

```{eval-rst}
.. autoclass:: pySPFM.SparseDeconvolution
   :members:
   :inherited-members:
```

## LowRankPlusSparse

```{eval-rst}
.. autoclass:: pySPFM.LowRankPlusSparse
   :members:
   :inherited-members:
```

## StabilitySelection

```{eval-rst}
.. autoclass:: pySPFM.StabilitySelection
   :members:
   :inherited-members:
```
