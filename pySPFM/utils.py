"""Utils of pySPFM."""
import logging
from os.path import expanduser, join

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


def dask_scheduler(jobs):
    """
    Checks if the user has a dask_jobqueue configuration file, and if so,
    returns the appropriate scheduler according to the file parameters
    """
    # look if default ~ .config/dask/jobqueue.yaml exists
    with open(join(expanduser("~"), ".config/dask/jobqueue.yaml"), "r") as stream:
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
        config.set(distributed__comm__timeouts__tcp="90s")
        config.set(distributed__comm__timeouts__connect="90s")
        config.set(scheduler="single-threaded")
        config.set({"distributed.scheduler.allowed-failures": 50})
        config.set(admin__tick__limit="3h")
        if "sge" in data["jobqueue"]:
            cluster = SGECluster()
            cluster.scale(jobs)
        elif "pbs" in data["jobqueue"]:
            cluster = PBSCluster()
            cluster.scale(jobs)
        elif "slurm" in data["jobqueue"]:
            cluster = SLURMCluster()
            cluster.scale(jobs)
        else:
            LGR.warning(
                "dask configuration wasn't detected, "
                "if you are using a cluster please look at "
                "the jobqueue YAML example, modify it so it works in your cluster "
                "and add it to ~/.config/dask "
                "local configuration will be used."
                "You can find a jobqueue YAML example in the pySPFM/jobqueue.yaml file."
            )
            cluster = None
    if cluster is None:
        client = None
    else:
        client = Client(cluster)
    return client, cluster
