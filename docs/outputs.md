# Outputs of pySPFM

## Outputs with a selected regularization parameter lambda

For single-echo data:

```{eval-rst}
+-------------------------------------------+---------------------------------------------------------------------------------------------------+
| Filename                                  | Content                                                                                           |
+-------------------------------------------+---------------------------------------------------------------------------------------------------+
| basename_pySPFM_activityInducing.nii.gz   | Estimated activity-inducing signal                                                                |
+-------------------------------------------+---------------------------------------------------------------------------------------------------+
| basename_pySPFM_denoised_bold.nii.gz      | Denoised BOLD signal (estimated activity-inducing signal convolved with the HRF)                  |
+-------------------------------------------+---------------------------------------------------------------------------------------------------+
| basename_pySPFM_innovation.nii.gz         | Estimated innovation signal (if using the block model)                                            |
+-------------------------------------------+---------------------------------------------------------------------------------------------------+
| basename_pySPFM_lambda.nii.gz             | Map of selected lambda values                                                                     |
+-------------------------------------------+---------------------------------------------------------------------------------------------------+
| basename_pySPFM_MAD.nii.gz                | Map of the mean absolute deviation of the residuals (estimated level of noise in original data)   |
+-------------------------------------------+---------------------------------------------------------------------------------------------------+
| _references.txt                           | References to the methods used in the analysis                                                    |
+-------------------------------------------+---------------------------------------------------------------------------------------------------+
| call.sh                                   | Command used to run the analysis                                                                  |
+-------------------------------------------+---------------------------------------------------------------------------------------------------+

```

For multi-echo data, the outputs are the same as for single-echo data, but with a file for each echo in the case of the denoised BOLD signal and the MAD.

## Outputs with stability selection

The outputs of the stability selection analysis are the same for single-echo and multi-echo data:

```{eval-rst}
+-------------------------------------------+---------------------------------------------------------------------------------------------------+
| Filename                                  | Content                                                                                           |
+-------------------------------------------+---------------------------------------------------------------------------------------------------+
| basename_pySPFM_AUC.nii.gz                | Estimated area under the curve (probability of containing a neuronal-related event)               |
+-------------------------------------------+---------------------------------------------------------------------------------------------------+
| _references.txt                           | References to the methods used in the analysis                                                    |
+-------------------------------------------+---------------------------------------------------------------------------------------------------+
| call.sh                                   | Command used to run the analysis                                                                  |
+-------------------------------------------+---------------------------------------------------------------------------------------------------+

```

## Outputs of the auc_to_estimates function

The outputs of the `auc_to_estimates` function are the same as for the single $\lambda$ analysis, with the addition of the following files:

```{eval-rst}
+-------------------------------------------+---------------------------------------------------------------------------------------------------+
| Filename                                  | Content                                                                                           |
+-------------------------------------------+---------------------------------------------------------------------------------------------------+
| basename_pySPFM_aucThresholded.nii.gz     | Map of the thresholded AUC values                                                                 |
+-------------------------------------------+---------------------------------------------------------------------------------------------------+

```
