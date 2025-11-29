"""Tests for pySPFM utils module."""

import logging
import os.path as op

import yaml

from pySPFM import utils


class TestSetupLoggers:
    """Tests for setup_loggers function."""

    def test_setup_loggers_basic(self):
        """Test basic logger setup without files."""
        utils.setup_loggers()

        lgr = logging.getLogger("GENERAL")
        assert lgr.level in [logging.INFO, logging.DEBUG, logging.WARNING]

        utils.teardown_loggers()

    def test_setup_loggers_quiet(self):
        """Test logger setup in quiet mode."""
        utils.setup_loggers(quiet=True)

        lgr = logging.getLogger("GENERAL")
        assert lgr.level == logging.WARNING

        utils.teardown_loggers()

    def test_setup_loggers_debug(self):
        """Test logger setup in debug mode."""
        utils.setup_loggers(debug=True)

        lgr = logging.getLogger("GENERAL")
        assert lgr.level == logging.DEBUG

        utils.teardown_loggers()

    def test_setup_loggers_with_files(self, testpath):
        """Test logger setup with log and reference files."""
        logname = op.join(testpath, "test.log")
        refname = op.join(testpath, "test_refs.txt")

        utils.setup_loggers(logname=logname, refname=refname)

        lgr = logging.getLogger("GENERAL")
        ref_lgr = logging.getLogger("REFERENCES")

        # Check handlers were added
        assert len(lgr.handlers) >= 2  # File handler + stream handler
        assert len(ref_lgr.handlers) >= 1  # File handler

        utils.teardown_loggers()


class TestTeardownLoggers:
    """Tests for teardown_loggers function."""

    def test_teardown_loggers(self):
        """Test that teardown removes all handlers."""
        utils.setup_loggers()

        lgr = logging.getLogger("GENERAL")
        ref_lgr = logging.getLogger("REFERENCES")

        utils.teardown_loggers()

        assert len(lgr.handlers) == 0
        assert len(ref_lgr.handlers) == 0


class TestGetOutname:
    """Tests for get_outname function."""

    def test_get_outname_no_bids(self):
        """Test output name without BIDS format."""
        result = utils.get_outname("output", "innovation", "nii.gz", use_bids=False)
        assert result == "output_pySPFM_innovation.nii.gz"

    def test_get_outname_with_bids(self):
        """Test output name with BIDS format."""
        result = utils.get_outname("output", "innovation", "nii.gz", use_bids=True)
        assert result == "output_desc-innovation.nii.gz"

    def test_get_outname_different_keywords(self):
        """Test with different keywords."""
        assert "bold" in utils.get_outname("out", "bold", "nii", use_bids=True)
        assert "lambda" in utils.get_outname("out", "lambda", "nii", use_bids=False)


class TestGetKeywordDescription:
    """Tests for get_keyword_description function."""

    def test_innovation_keyword(self):
        """Test description for innovation keyword."""
        desc = utils.get_keyword_description("innovation_signal")
        assert "innovation" in desc.lower() and "derivative" in desc.lower()

    def test_beta_keyword(self):
        """Test description for beta keyword."""
        desc = utils.get_keyword_description("beta_map")
        assert "activity-inducing" in desc.lower()

    def test_activity_inducing_keyword(self):
        """Test description for activityInducing keyword."""
        desc = utils.get_keyword_description("activityInducing")
        assert "R2*" in desc and "activity-inducing" in desc.lower()

    def test_bold_keyword(self):
        """Test description for bold keyword."""
        desc = utils.get_keyword_description("bold_signal")
        assert "BOLD" in desc and "denoised" in desc.lower()

    def test_lambda_keyword(self):
        """Test description for lambda keyword."""
        desc = utils.get_keyword_description("lambda_map")
        assert "regularization" in desc.lower()

    def test_mad_keyword(self):
        """Test description for MAD keyword."""
        desc = utils.get_keyword_description("MAD_estimate")
        assert "noise" in desc.lower() and "deviation" in desc.lower()


class TestDaskScheduler:
    """Tests for dask_scheduler function."""

    def test_dask_scheduler_no_jobqueue(self):
        """Test dask_scheduler without jobqueue file."""
        client, cluster = utils.dask_scheduler(jobs=2, jobqueue=None)

        assert cluster is None
        assert client is None

    def test_dask_scheduler_invalid_jobqueue(self, testpath):
        """Test dask_scheduler with invalid jobqueue type."""
        # Create a jobqueue file with unknown type
        jobqueue_path = op.join(testpath, "invalid_jobqueue.yaml")
        with open(jobqueue_path, "w") as f:
            yaml.dump({"jobqueue": {"unknown": {}}}, f)

        client, cluster = utils.dask_scheduler(jobs=2, jobqueue=jobqueue_path)

        # Should return None when type is not recognized
        assert cluster is None
        assert client is None


class TestInitiateCluster:
    """Tests for initiate_cluster function."""

    def test_initiate_cluster_unknown_type(self):
        """Test initiate_cluster with unknown cluster type."""
        data = {"jobqueue": {"unknown_type": {}}}
        result = utils.initiate_cluster(data, jobs=2)
        assert result is None
