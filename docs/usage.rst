#####
Usage
#####

`pySPFM` has two main ways of usage: using a fixed selection of the regularization parameter lambda,
or using the stability selection method, which avoids the selection of lambda and yields more robust
estimates of the neuronal-related signal at the cost of higher computation time.

##### Fixed lambda

The fixed lambda method is the simplest one. It consists in choosing one of the following methods
to automatically calculate the regularization parameter lambda:

- `universal threshold`:
- `lower universal threshold`:
- `median absolute deviation`:
- `updating median absolute deviation`:
- `percentage of maximum lambda`:
- `factor of median absolute deviation`:

.. code-block:: bash

    pySPFM -i my_echo_1.nii.gz my_echo_3.nii.gz my_echo_3.nii.gz -m my_mask.nii.gz
    -te 14.5, 38.5, 62.5 -o my_subject_pySPFM -tr 2 -d my_results_directory -crit mad

##### Stability selection

The stability selection procedure solves the regularization problem in a number of subsampled
surrogate datasets with the Least Angle Regression algorithm. 

.. code-block:: bash

    pySPFM -i my_data.nii.gz -m my_mask.nii.gz -o my_subject_stability_selection -tr 2
    -d my_results_directory -crit stability -nsur 50