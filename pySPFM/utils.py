"""Utils of pySPFM."""
import logging

LGR = logging.getLogger("GENERAL")
RefLGR = logging.getLogger("REFERENCES")


def setup_loggers(logname=None, refname=None, quiet=False, debug=False):
    """Set up loggers.

    Parameters
    ----------
    logname : str, optional
        Name of the log file, by default None
    refname : str, optional
        Name of the reference file, by default None
    quiet : bool, optional
        Whether the logger should run in quiet mode, by default False
    debug : bool, optional
        Whether the logger should run in debug mode, by default False
    """
    # Set up the general logger
    log_formatter = logging.Formatter(
        "%(asctime)s\t%(module)s.%(funcName)-12s\t%(levelname)-8s\t%(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    stream_formatter = logging.Formatter(
        "%(levelname)-8s %(module)s:%(funcName)s:%(lineno)d %(message)s"
    )
    # set up general logging file and open it for writing
    if logname:
        log_handler = logging.FileHandler(logname)
        log_handler.setFormatter(log_formatter)
        LGR.addHandler(log_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_formatter)
    LGR.addHandler(stream_handler)

    if quiet:
        LGR.setLevel(logging.WARNING)
    elif debug:
        LGR.setLevel(logging.DEBUG)
    else:
        LGR.setLevel(logging.INFO)

    # Loggers for references
    text_formatter = logging.Formatter("%(message)s")

    if refname:
        ref_handler = logging.FileHandler(refname)
        ref_handler.setFormatter(text_formatter)
        RefLGR.setLevel(logging.INFO)
        RefLGR.addHandler(ref_handler)
        RefLGR.propagate = False


def teardown_loggers():
    """Remove logger handler."""
    for local_logger in (RefLGR, LGR):
        for handler in local_logger.handlers[:]:
            handler.close()
            local_logger.removeHandler(handler)


def get_outname(outname, keyword, ext, use_bids=False):
    """Get the output name.

    Parameters
    ----------
    outname : str
        Name of the output file.
    keyword : str
        Keyword added by pySPFM.
    ext : str
        Extension of the output file.
    use_bids : bool, optional
        Whether the output file is in BIDS format, by default False

    Returns
    -------
    outname : str
        Name of the output file.
    """
    if use_bids:
        outname = f"{outname}_desc-{keyword}.{ext}"
    else:
        outname = f"{outname}_pySPFM_{keyword}.{ext}"
    return outname


def get_keyword_description(keyword):
    """
    Get the description of the keyword for BIDS sidecar


    Parameters
    ----------
    keyword : str
        Keyword added by pySPFM.

    Returns
    -------
    keyword_description : str
        Description of the keyword.
    """

    if "innovation" in keyword:
        keyword_description = (
            "Deconvolution-estimated innovation signal; i.e., the derivative"
            "of the activity-inducing signal."
        )
    elif "beta" in keyword:
        keyword_description = (
            "Deconvolution-estimated activity-inducing signal; i.e., induces BOLD response."
        )
    elif "activityInducing" in keyword:
        keyword_description = (
            "Deconvolution-estimated activity-inducing signal that represents"
            "changes in the R2* component of the multi-echo signal; i.e., induces BOLD response."
        )
    elif "bold" in keyword:
        keyword_description = (
            "Deconvolution-denoised activity-related signal; i.e., denoised BOLD signal."
        )
    elif "lambda" in keyword:
        keyword_description = (
            "Map of the regularization parameter lambda used to solve the deconvolution problem."
        )
    elif "MAD" in keyword:
        keyword_description = (
            "Estimated mean absolute deviation of the noise; i.e., noise level"
            "of the signal to be deconvolved."
        )

    return keyword_description
