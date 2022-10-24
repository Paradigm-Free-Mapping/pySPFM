import logging

import nibabel as nib
import numpy as np

from pySPFM.io import read_data

LGR = logging.getLogger("GENERAL")


def threshold_auc(auc_fn, mask_fn, thr, thr_strategy, n_scans):

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
    # If thr is 0, then the AUC is supposed to be already thresholded
    else:
        LGR.warning("Threshold 0 selected. AUC is assumed to be already thresholded.")
        auc_thr = auc

    LGR.info("AUC data thresholded.")

    return auc_thr
