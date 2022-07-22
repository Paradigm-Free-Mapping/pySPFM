"""HRF Matrix file."""
import logging
import subprocess

import numpy as np
from nilearn.glm.first_level import glover_hrf, spm_hrf

LGR = logging.getLogger("GENERAL")
RefLGR = logging.getLogger("REFERENCES")


class HRFMatrix:
    """A class for generating an HRF matrix.

    Parameters
    ----------
    TE : list
        Values of TE in ms, by default None
    model : str
        Model to use for HRF, by default "spm"
    block : bool
        Whether to use the block model in favor of the spike model, by default false
    custom : str
        Path to custom HRF file, by default None
    """

    def __init__(
        self,
        te=None,
        model="spm",
        block=True,
    ):
        self.te = te
        self.model = model
        self.block = block

    def generate_hrf(self, tr, n_scans):
        """Generate HRF matrix.

        Parameters
        ----------
        tr : float
            tr of the acquisition.
        n_scans : int
            Number of scans.

        Returns
        -------
        self.hrf_ : array_like
            A hemodynamic response function (HRF).
        """

        # Read custom HRF from file if self.model ends in .1D or .txt
        if self.model.endswith(".1D") or self.model.endswith(".txt"):
            hrf = np.loadtxt(self.custom)
        else:
            # Get HRF from nilearn
            if self.model == "spm":
                hrf = spm_hrf(tr, oversampling=1, time_length=n_scans * tr)
            elif self.model == "glover":
                hrf = glover_hrf(tr, oversampling=1, time_length=n_scans * tr)

        # Calculate maximum HRF value
        max_val = max(abs(hrf))

        # Generate HRF matrix
        hrf_mtx = hrf
        for i in range(n_scans - 1):
            foo = np.append(np.zeros(i + 1), hrf[0 : (len(hrf) - i - 1)])
            hrf_mtx = np.column_stack((hrf_mtx, foo))

        # Normalize HRF matrix
        hrf_mtx = hrf_mtx / max_val

        # Concatenate and scale HRFs for multi-echo,
        # leave it as it is for single-echo.
        if len(self.te) > 1:
            # Add integrator if necessary
            if self.block:
                hrf_mtx_te = -self.te[0] * np.dot(hrf_mtx, np.tril(np.ones(n_scans)))
            else:
                hrf_mtx_te = -self.te[0] * hrf_mtx

            # Concatenate and scale HRFs for multi-echo
            for teidx in range(len(self.te) - 1):
                # Add integrator if necessary
                if self.block:
                    hrf_mtx_te = np.vstack(
                        (
                            hrf_mtx_te,
                            -self.te[teidx + 1] * np.dot(hrf_mtx, np.tril(np.ones(n_scans))),
                        )
                    )
                else:
                    hrf_mtx_te = np.vstack((hrf_mtx_te, -self.te[teidx + 1] * hrf_mtx))

            self.hrf_ = hrf_mtx_te
        else:
            if self.block:
                self.hrf_ = np.dot(hrf_mtx, np.tril(np.ones(n_scans)))
            else:
                self.hrf_ = hrf_mtx

        return self

    def hrf_afni(tr, hrf_model="SPMG1"):
        """Generate HRF with AFNI's 3dDeconvolve.

        Parameters
        ----------
        tr : float
            tr of the acquisition.
        hrf_model : str
            3dDeconvolve option to select HRF shape, by default "SPMG1"

        Returns
        -------
        hrf : array_like
            A hemodynamic response function (HRF).

        Notes
        -----
        AFNI installation is needed as it runs 3dDeconvolve on the terminal with subprocess.
        """
        dur_hrf = 8
        last_hrf_sample = 1
        # Increases duration until last HRF sample is zero
        while last_hrf_sample != 0:
            dur_hrf = 2 * dur_hrf
            hrf_command = (
                "3dDeconvolve -x1D_stop -nodata %d %f -polort -1 -num_stimts 1 -stim_times 1 "
                "'1D:0' '%s' -quiet -x1D stdout: | 1deval -a stdin: -expr 'a'"
            ) % (dur_hrf, tr, hrf_model)
            hrf_tr_str = subprocess.check_output(
                hrf_command, shell=True, universal_newlines=True
            ).splitlines()
            hrf = np.array([float(i) for i in hrf_tr_str])
            last_hrf_sample = hrf[len(hrf) - 1]
            if last_hrf_sample != 0:
                LGR.info(
                    "Duration of HRF was not sufficient for specified model. Doubling duration "
                    "and computing again."
                )

        # Removes tail of zero samples
        while last_hrf_sample == 0:
            hrf = hrf[0 : len(hrf) - 1]
            last_hrf_sample = hrf[len(hrf) - 1]

        return hrf
