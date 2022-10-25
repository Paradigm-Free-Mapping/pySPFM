import logging

import nibabel as nib
import numpy as np

from pySPFM.io import read_data

LGR = logging.getLogger("GENERAL")


def circular_shift(auc, n_perm=1000, cutoff=0.05):

    # Create null distribution with circle shift method
    surr_data = np.zeros((len(auc), n_perm))
    for i in range(n_perm):
        surr_data[:, i] = np.roll(auc, np.random.randint(len(auc)))

    # For each TR, calculate if the AUC value is significant
    p_value = np.zeros(len(auc))
    surr_data_flat = surr_data.flatten()
    for tr in range(auc.shape[0]):
        p_value[tr] = np.mean(surr_data_flat >= auc[tr])

    # Return AUC with significant values
    auc_sig = auc.copy()
    auc_sig[p_value > cutoff] = 0

    return auc_sig


def threshold_auc(auc_fn, mask_fn, thr, thr_strategy, n_scans, cutoff=0.05):

    LGR.info("Reading AUC data...")
    auc, masker = read_data(auc_fn, mask_fn[0])
    LGR.info("AUC data read.")

    # Threshold the AUC if thr is not 0 and mask_fn has two elements
    if thr != 0 and len(mask_fn) == 2:
        # Read the second mask
        auc_mask = nib.load(mask_fn[1])

        # If the mask is 3D, then it is a binary mask or a static threshold
        if len(auc_mask.shape) == 3:
            # If the mask is binary, then read the AUC values inside of the mask
            if np.max(auc_mask.get_fdata()) == 1:

                auc_thr_values = read_data(auc_fn, mask_fn[1])[0]

                if thr_strategy == "static":
                    LGR.info(
                        f"Thresholding AUC values with a {thr}th percentile static threshold..."
                    )
                    # Threshold the whole-brain AUC based on the thr percentile of the AUC values
                    # in the mask
                    auc_thr = auc - np.percentile(auc_thr_values, thr)
                    auc_thr[auc_thr < 0] = 0
                else:
                    LGR.info(
                        f"Thresholding AUC values with a {thr}th percentile time-dependet "
                        "threshold..."
                    )

                    # Calculate and apply percentile at each TR
                    auc_thr = np.zeros(auc.shape)
                    for tr_idx in range(n_scans):
                        auc_thr[tr_idx, :] = auc[tr_idx, :] - np.percentile(
                            auc_thr_values[tr_idx, :], thr
                        )
                        auc_thr[tr_idx, auc_thr[tr_idx, :] < 0] = 0

            # If the mask is a static threshold, then apply it to the AUC values
            else:
                LGR.info("Thresholding AUC values based on the given 3D threshold...")
                auc_mask_data = masker.fit_transform(auc_mask)

                # Threshold the AUC values
                auc_thr = auc - auc_mask_data
                auc_thr[auc_thr < 0] = 0

        # If the mask is 4D, then it is a time-dependent threshold
        elif len(auc_mask.shape) == 4:
            LGR.info("Thresholding AUC values based on the given 4D threshold...")
            # Read the time-dependent threshold
            auc_mask_data = masker.fit_transform(auc_mask)

            # Threshold the AUC
            auc_thr = auc - auc_mask_data
            auc_thr[auc_thr < 0] = 0
        else:
            raise ValueError("The mask used to threshold the AUC must be 3D or 4D.")

    # Raise error if thr is not 0 and mask_fn has only one element
    elif thr != 0 and len(mask_fn) != 1:
        raise ValueError("If the threshold is not 0, then the 'mask' flag must have two elements.")

    # If thr is 0, but thr_strategy is 'circular', then apply the circular shift method
    elif thr_strategy == "circular":
        LGR.info("Thresholding AUC values with the circular shift method...")
        auc_thr = np.zeros(auc.shape)
        for voxel_idx in range(auc.shape[1]):
            auc_thr[:, voxel_idx] = circular_shift(auc[:, voxel_idx], cutoff=cutoff)

    # If thr is 0, then the AUC is supposed to be already thresholded
    else:
        LGR.warning("Threshold 0 selected. AUC is assumed to be already thresholded.")
        auc_thr = auc

    LGR.info("AUC data thresholded.")

    return auc_thr
