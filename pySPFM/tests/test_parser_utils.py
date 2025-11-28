"""Tests for pySPFM parser_utils module."""

import argparse
import os.path as op

import numpy as np
import pytest

from pySPFM.workflows import parser_utils


class TestCheckHrfValue:
    """Tests for check_hrf_value function."""

    def test_spm_model(self):
        """Test that 'spm' is accepted."""
        result = parser_utils.check_hrf_value("spm")
        assert result == "spm"

    def test_glover_model(self):
        """Test that 'glover' is accepted."""
        result = parser_utils.check_hrf_value("glover")
        assert result == "glover"

    def test_custom_1d_file(self, testpath):
        """Test that valid .1D file is accepted."""
        hrf_path = op.join(testpath, "custom.1D")
        np.savetxt(hrf_path, np.array([0.0, 0.5, 1.0, 0.5, 0.0]))

        result = parser_utils.check_hrf_value(hrf_path)
        assert result == hrf_path

    def test_custom_txt_file(self, testpath):
        """Test that valid .txt file is accepted."""
        hrf_path = op.join(testpath, "custom.txt")
        np.savetxt(hrf_path, np.array([0.0, 0.5, 1.0, 0.5, 0.0]))

        result = parser_utils.check_hrf_value(hrf_path)
        assert result == hrf_path

    def test_invalid_model(self):
        """Test that invalid model raises ArgumentTypeError."""
        with pytest.raises(argparse.ArgumentTypeError, match="HRF model must be"):
            parser_utils.check_hrf_value("invalid")

    def test_nonexistent_file(self):
        """Test that non-existent file raises error."""
        with pytest.raises((argparse.ArgumentTypeError, SystemExit)):
            parser_utils.check_hrf_value("/nonexistent/path.1D")

    def test_check_hrf_value_no_parser(self, testpath):
        """Test check_hrf_value with is_parser=False."""
        # With is_parser=False, it should raise for non-spm/glover strings
        # that don't have .1D or .txt extension
        with pytest.raises(argparse.ArgumentTypeError):
            parser_utils.check_hrf_value("invalid", is_parser=False)


class TestIsValidFile:
    """Tests for is_valid_file function."""

    def test_valid_file(self, testpath):
        """Test with valid existing file."""
        file_path = op.join(testpath, "valid_file.txt")
        with open(file_path, "w") as f:
            f.write("test")

        parser = argparse.ArgumentParser()
        result = parser_utils.is_valid_file(parser, file_path)
        assert result == file_path

    def test_invalid_file(self):
        """Test with non-existent file."""
        parser = argparse.ArgumentParser()

        with pytest.raises(SystemExit):
            parser_utils.is_valid_file(parser, "/nonexistent/file.txt")

    def test_none_argument(self):
        """Test with None argument - should return None without error."""
        parser = argparse.ArgumentParser()
        result = parser_utils.is_valid_file(parser, None)
        assert result is None
