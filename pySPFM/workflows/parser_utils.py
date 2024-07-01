"""Functions for parsers."""

import argparse
import os.path as op


def check_hrf_value(string, is_parser=True):
    """
    Check if argument is 'spm' or 'glover'.

    If it isn't, check if it is an existing '.1D' or '.txt' file.
    """
    if string == "spm" or string == "glover":
        return string
    elif (string.endswith(".1D") or string.endswith(".txt")) and is_parser:
        return is_valid_file(argparse.ArgumentParser(), string)
    else:
        raise argparse.ArgumentTypeError(
            "HRF model must be 'spm', 'glover', or a custom HRF file with the '.1D' "
            "or '.txt' extension."
        )


def is_valid_file(parser, arg):
    """
    Check if argument is existing file.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Argument parser.
    arg : str
        Argument to check.

    Returns
    -------
    arg : str
        Argument if it is an existing file.
    """
    if not op.isfile(arg) and arg is not None:
        parser.error(f"The file {arg} does not exist!")

    return arg
