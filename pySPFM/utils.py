"""Utils of pySPFM."""

import logging

import yaml
from dask import config
from dask.distributed import Client
from dask_jobqueue import PBSCluster, SGECluster, SLURMCluster

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
    Get the description of the keyword for BIDS sidecar.

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
    elif "auc" in keyword.lower():
        keyword_description = (
            "Area under the curve of the deconvolution-estimated activity-inducing signal."
        )
    else:
        keyword_description = f"pySPFM output: {keyword}"

    return keyword_description


def dask_scheduler(jobs, jobqueue=None):
    """
    Check if the user has a dask_jobqueue configuration file.

    If so, return the appropriate scheduler according to the file parameters.

    Parameters
    ----------
    jobs : int
        Number of jobs.
    jobqueue : str, optional
        Path to the jobqueue YAML file, by default None

    Returns
    -------
    client : dask.distributed.Client
        Dask client.
    cluster : dask.distributed.Cluster
        Dask cluster.
    """
    # look if jobqueue.yaml exists
    if jobqueue is None:
        data = None
    else:
        LGR.info(f"Using jobqueue configuration file: {jobqueue}")
        with open(jobqueue) as stream:
            data = yaml.load(stream, Loader=yaml.FullLoader)

    if data is None:
        LGR.warning(
            "dask configuration wasn't detected, "
            "if you are using a cluster please look at "
            "the jobqueue YAML example, modify it so it works in your cluster "
            "and add it to ~/.config/dask "
            "local configuration will be used."
            "You can find a jobqueue YAML example in the pySPFM/jobqueue.yaml file."
        )
        cluster = None
    else:
        cluster = initiate_cluster(data, jobs)
    client = None if cluster is None else Client(cluster)
    return client, cluster


def initiate_cluster(data, jobs):
    """
    Initiate a dask cluster.

    Parameters
    ----------
    data : dict
        Dictionary with the jobqueue parameters.
    jobs : int
        Number of jobs.

    Returns
    -------
    result : dask.distributed.Cluster
        Dask cluster.
    """
    config.set(distributed__comm__timeouts__tcp="90s")
    config.set(distributed__comm__timeouts__connect="90s")
    config.set(scheduler="single-threaded")
    config.set({"distributed.scheduler.allowed-failures": 50})
    config.set(admin__tick__limit="3h")
    if "sge" in data["jobqueue"]:
        result = SGECluster()
        result.scale(jobs)
    elif "pbs" in data["jobqueue"]:
        result = PBSCluster()
        result.scale(jobs)
    elif "slurm" in data["jobqueue"]:
        result = SLURMCluster()
        result.scale(jobs)
    else:
        LGR.warning(
            "dask configuration wasn't detected, "
            "if you are using a cluster please look at "
            "the jobqueue YAML example, modify it so it works in your cluster "
            "and add it to ~/.config/dask "
            "local configuration will be used."
            "You can find a jobqueue YAML example in the pySPFM/jobqueue.yaml file."
        )
        result = None
    return result
