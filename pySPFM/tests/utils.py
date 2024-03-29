"""Utility functions for pySPFM's test suite."""

import os.path as op


def get_tests_resource_path():
    """Return the path to general resources, terminated with separator.

    Resources are kept outside package folder in "datasets".
    Based on function by Yaroslav Halchenko used in Neurosynth Python package.
    """
    return op.abspath(op.join(op.dirname(__file__), "data") + op.sep)
