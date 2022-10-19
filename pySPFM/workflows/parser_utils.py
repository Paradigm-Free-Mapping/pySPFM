"""
Functions for parsers.
"""
import argparse
import os.path as op


def check_hrf_value(string, is_parser=True):
    """
    Check if argument is 'spm' or 'glover',
    or is an existing '.1D' or '.txt' file.
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


def check_thr_value(arg, is_parser=True):
    """
    Check if argument is float or str.
    """
    if isinstance(arg, float) and arg >= 0:
        return arg
    elif isinstance(arg, float) and arg < 0:
        raise argparse.ArgumentTypeError("Threshold must be a float >=0.")
    elif isinstance(arg, str) and is_parser:
        return is_valid_file(argparse.ArgumentParser(), arg)
    else:
        raise argparse.ArgumentTypeError("Threshold must be a float or a filename.")


def is_valid_file(parser, arg):
    """
    Check if argument is existing file.
    """
    if not op.isfile(arg) and arg is not None:
        parser.error("The file {0} does not exist!".format(arg))

    return arg
