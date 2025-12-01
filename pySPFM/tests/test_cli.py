"""Tests for pySPFM CLI to improve coverage."""

import os
import tempfile

import numpy as np
import pytest


class TestCLIParser:
    """Tests for CLI argument parser."""

    def test_get_parser(self):
        """Test that parser is created correctly."""
        from pySPFM.cli.run import _get_parser

        parser = _get_parser()

        assert parser is not None
        assert parser.prog == "pySPFM"

    def test_parser_version(self, capsys):
        """Test version argument."""
        from pySPFM.cli.run import _get_parser

        parser = _get_parser()

        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])

        assert exc_info.value.code == 0

    def test_sparse_subcommand_args(self):
        """Test sparse subcommand argument parsing."""
        from pySPFM.cli.run import _get_parser

        parser = _get_parser()
        args = parser.parse_args(
            [
                "sparse",
                "-i",
                "data.nii.gz",
                "-m",
                "mask.nii.gz",
                "-o",
                "output",
                "--tr",
                "2.0",
                "--criterion",
                "bic",
            ]
        )

        assert args.command == "sparse"
        assert args.data == ["data.nii.gz"]
        assert args.mask == "mask.nii.gz"
        assert args.out_prefix == "output"
        assert args.tr == 2.0
        assert args.criterion == "bic"

    def test_stability_subcommand_args(self):
        """Test stability subcommand argument parsing."""
        from pySPFM.cli.run import _get_parser

        parser = _get_parser()
        args = parser.parse_args(
            [
                "stability",
                "-i",
                "data.nii.gz",
                "-m",
                "mask.nii.gz",
                "-o",
                "output",
                "--tr",
                "2.0",
                "--n-surrogates",
                "30",
            ]
        )

        assert args.command == "stability"
        assert args.n_surrogates == 30

    def test_lowrank_subcommand_args(self):
        """Test lowrank subcommand argument parsing."""
        from pySPFM.cli.run import _get_parser

        parser = _get_parser()
        args = parser.parse_args(
            [
                "lowrank",
                "-i",
                "data.nii.gz",
                "-m",
                "mask.nii.gz",
                "-o",
                "output",
                "--tr",
                "2.0",
                "--eigval-threshold",
                "0.2",
            ]
        )

        assert args.command == "lowrank"
        assert args.eigval_threshold == 0.2

    def test_optional_args_defaults(self):
        """Test default values for optional arguments."""
        from pySPFM.cli.run import _get_parser

        parser = _get_parser()
        args = parser.parse_args(
            [
                "sparse",
                "-i",
                "data.nii.gz",
                "-m",
                "mask.nii.gz",
                "-o",
                "output",
                "--tr",
                "2.0",
            ]
        )

        assert args.out_dir == "."
        assert args.n_jobs == 1
        assert args.criterion == "bic"
        assert args.max_iter == 400
        assert args.tol == 1e-6
        assert args.group == 0.0


class TestCLIHelpers:
    """Tests for CLI helper functions."""

    def test_save_call_script(self):
        """Test _save_call_script function."""
        from pySPFM.cli.run import _save_call_script

        with tempfile.TemporaryDirectory() as tmpdir:
            argv = ["pySPFM", "sparse", "-i", "data.nii.gz"]
            _save_call_script(tmpdir, argv)

            call_file = os.path.join(tmpdir, "call.sh")
            assert os.path.exists(call_file)

            with open(call_file) as f:
                content = f.read()
                assert "#!/bin/bash" in content
                assert "pySPFM sparse -i data.nii.gz" in content

    def test_compute_mad(self):
        """Test _compute_mad function."""
        from pySPFM.cli.run import _compute_mad

        np.random.seed(42)
        n_timepoints = 50
        n_voxels = 10

        # Create simple test data
        hrf_matrix = np.eye(n_timepoints)
        coef = np.random.randn(n_timepoints, n_voxels)
        y = np.dot(hrf_matrix, coef) + np.random.randn(n_timepoints, n_voxels) * 0.1

        mad = _compute_mad(y, hrf_matrix, coef)

        assert mad.shape == (n_voxels,)
        assert np.all(mad >= 0)


class TestMainNoCommand:
    """Test main function without command."""

    def test_main_no_command(self, capsys):
        """Test that main prints help when no command given."""
        from pySPFM.cli.run import main

        with pytest.raises(SystemExit) as exc_info:
            main(["pySPFM"])

        assert exc_info.value.code == 0

    def test_main_help(self, capsys):
        """Test that main prints help with --help."""
        from pySPFM.cli.run import main

        with pytest.raises(SystemExit) as exc_info:
            main(["pySPFM", "--help"])

        assert exc_info.value.code == 0


class TestCLIMultiEchoArgs:
    """Tests for multi-echo CLI arguments."""

    def test_multi_echo_data_args(self):
        """Test parsing multiple input files for multi-echo."""
        from pySPFM.cli.run import _get_parser

        parser = _get_parser()
        args = parser.parse_args(
            [
                "sparse",
                "-i",
                "echo1.nii.gz",
                "echo2.nii.gz",
                "echo3.nii.gz",
                "-m",
                "mask.nii.gz",
                "-o",
                "output",
                "--tr",
                "2.0",
                "-te",
                "14.5",
                "29.0",
                "43.5",
            ]
        )

        assert len(args.data) == 3
        assert args.te == [14.5, 29.0, 43.5]

    def test_regressors_arg(self):
        """Test regressors argument parsing."""
        from pySPFM.cli.run import _get_parser

        parser = _get_parser()
        args = parser.parse_args(
            [
                "sparse",
                "-i",
                "data.nii.gz",
                "-m",
                "mask.nii.gz",
                "-o",
                "output",
                "--tr",
                "2.0",
                "--regressors",
                "confounds.txt",
            ]
        )

        assert args.regressors == "confounds.txt"

    def test_bids_flag(self):
        """Test BIDS naming flag."""
        from pySPFM.cli.run import _get_parser

        parser = _get_parser()
        args = parser.parse_args(
            [
                "sparse",
                "-i",
                "data.nii.gz",
                "-m",
                "mask.nii.gz",
                "-o",
                "output",
                "--tr",
                "2.0",
                "--bids",
            ]
        )

        assert args.use_bids is True

    def test_block_model_flag(self):
        """Test block model flag."""
        from pySPFM.cli.run import _get_parser

        parser = _get_parser()
        args = parser.parse_args(
            [
                "sparse",
                "-i",
                "data.nii.gz",
                "-m",
                "mask.nii.gz",
                "-o",
                "output",
                "--tr",
                "2.0",
                "--block",
            ]
        )

        assert args.block_model is True
