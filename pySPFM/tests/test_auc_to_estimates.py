"""Tests for auc_to_estimates workflow."""

import os.path as op

import pytest

from pySPFM.workflows import auc_to_estimates


class TestGetParser:
    """Tests for _get_parser function."""

    def test_parser_defaults(self):
        """Test parser default values."""
        parser = auc_to_estimates._get_parser()
        # Parse with minimal required args (using fake paths for testing parser structure)
        # We can't actually parse without valid files, so just test parser exists
        assert parser is not None
        assert parser.description is None  # No description set

    def test_parser_has_required_arguments(self):
        """Test parser has all required arguments."""
        parser = auc_to_estimates._get_parser()

        # Get all actions
        actions = {action.dest: action for action in parser._actions}

        # Check required arguments exist
        assert "data_fn" in actions
        assert "auc_fn" in actions
        assert "mask_fn" in actions
        assert "output_filename" in actions
        assert "tr" in actions

    def test_parser_has_optional_arguments(self):
        """Test parser has all optional arguments."""
        parser = auc_to_estimates._get_parser()

        # Get all actions
        actions = {action.dest: action for action in parser._actions}

        # Check optional arguments exist
        assert "thr" in actions
        assert "thr_strategy" in actions
        assert "out_dir" in actions
        assert "te" in actions
        assert "hrf_model" in actions
        assert "block_model" in actions
        assert "n_jobs" in actions
        assert "use_bids" in actions
        assert "group" in actions
        assert "group_distance" in actions
        assert "block_dist" in actions
        assert "debug" in actions
        assert "quiet" in actions

    def test_parser_default_values(self):
        """Test parser default values are correct."""
        parser = auc_to_estimates._get_parser()

        # Get all actions
        actions = {action.dest: action for action in parser._actions}

        # Check default values
        assert actions["thr"].default == 95.0
        assert actions["thr_strategy"].default == "static"
        assert actions["out_dir"].default == "."
        assert actions["te"].default == [0]
        assert actions["hrf_model"].default == "spm"
        assert actions["block_model"].default is False
        assert actions["n_jobs"].default == 4
        assert actions["use_bids"].default is False
        assert actions["group"].default is False
        assert actions["group_distance"].default == 3
        assert actions["block_dist"].default == 2
        assert actions["debug"].default is False
        assert actions["quiet"].default is False

    def test_parser_thr_strategy_choices(self):
        """Test parser thr_strategy has correct choices."""
        parser = auc_to_estimates._get_parser()
        actions = {action.dest: action for action in parser._actions}

        assert actions["thr_strategy"].choices == ["static", "time"]


class TestAucToEstimates:
    """Tests for auc_to_estimates function."""

    def test_auc_to_estimates_single_echo_spike(
        self, nih_five_echo_1, mask_five_echo, test_AUC, testpath
    ):
        """Test auc_to_estimates with single echo data and spike model."""
        out_dir = op.join(testpath, "auc_test_single_spike")

        auc_to_estimates.auc_to_estimates(
            data_fn=[nih_five_echo_1],
            auc_fn=test_AUC,
            mask_fn=[mask_five_echo],
            output_filename="test_auc",
            tr=2.0,
            thr=0,  # No thresholding
            out_dir=out_dir,
            te=[0],
            hrf_model="spm",
            block_model=False,
            n_jobs=1,
            use_bids=False,
            quiet=True,
        )

        # Check output files exist (non-BIDS uses _pySPFM_ prefix)
        assert op.exists(op.join(out_dir, "test_auc_pySPFM_aucThresholded.nii.gz"))
        assert op.exists(op.join(out_dir, "test_auc_pySPFM_activityInducing.nii.gz"))
        assert op.exists(op.join(out_dir, "test_auc_pySPFM_denoised_bold.nii.gz"))

    def test_auc_to_estimates_single_echo_block(
        self, nih_five_echo_1, mask_five_echo, test_AUC, testpath
    ):
        """Test auc_to_estimates with single echo data and block model."""
        out_dir = op.join(testpath, "auc_test_single_block")

        auc_to_estimates.auc_to_estimates(
            data_fn=[nih_five_echo_1],
            auc_fn=test_AUC,
            mask_fn=[mask_five_echo],
            output_filename="test_auc",
            tr=2.0,
            thr=0,  # No thresholding
            out_dir=out_dir,
            te=[0],
            hrf_model="spm",
            block_model=True,
            n_jobs=1,
            use_bids=False,
            quiet=True,
        )

        # Check output files exist (non-BIDS uses _pySPFM_ prefix)
        assert op.exists(op.join(out_dir, "test_auc_pySPFM_aucThresholded.nii.gz"))
        assert op.exists(op.join(out_dir, "test_auc_pySPFM_activityInducing.nii.gz"))
        assert op.exists(op.join(out_dir, "test_auc_pySPFM_denoised_bold.nii.gz"))

    def test_auc_to_estimates_with_bids(self, nih_five_echo_1, mask_five_echo, test_AUC, testpath):
        """Test auc_to_estimates with BIDS output."""
        out_dir = op.join(testpath, "auc_test_bids")

        auc_to_estimates.auc_to_estimates(
            data_fn=[nih_five_echo_1],
            auc_fn=test_AUC,
            mask_fn=[mask_five_echo],
            output_filename="sub-01_task-rest",
            tr=2.0,
            thr=0,
            out_dir=out_dir,
            te=[0],
            hrf_model="spm",
            block_model=False,
            n_jobs=1,
            use_bids=True,
            quiet=True,
        )

        # Check BIDS output files exist (uses _desc- prefix)
        assert op.exists(op.join(out_dir, "sub-01_task-rest_desc-aucThresholded.nii.gz"))
        assert op.exists(op.join(out_dir, "sub-01_task-rest_desc-activityInducing.nii.gz"))
        assert op.exists(op.join(out_dir, "sub-01_task-rest_desc-denoised_bold.nii.gz"))
        # Check JSON sidecar exists
        assert op.exists(op.join(out_dir, "dataset_description.json"))

    def test_auc_to_estimates_with_grouping(
        self, nih_five_echo_1, mask_five_echo, test_AUC, testpath
    ):
        """Test auc_to_estimates with grouping enabled."""
        out_dir = op.join(testpath, "auc_test_group")

        auc_to_estimates.auc_to_estimates(
            data_fn=[nih_five_echo_1],
            auc_fn=test_AUC,
            mask_fn=[mask_five_echo],
            output_filename="test_auc",
            tr=2.0,
            thr=0,
            out_dir=out_dir,
            te=[0],
            hrf_model="spm",
            block_model=False,
            n_jobs=1,
            use_bids=False,
            group=True,
            group_distance=3,
            quiet=True,
        )

        # Check output files exist (non-BIDS uses _pySPFM_ prefix)
        assert op.exists(op.join(out_dir, "test_auc_pySPFM_aucThresholded.nii.gz"))
        assert op.exists(op.join(out_dir, "test_auc_pySPFM_activityInducing.nii.gz"))

    def test_auc_to_estimates_with_command_str(
        self, nih_five_echo_1, mask_five_echo, test_AUC, testpath
    ):
        """Test auc_to_estimates saves command string to file."""
        out_dir = op.join(testpath, "auc_test_cmd")

        auc_to_estimates.auc_to_estimates(
            data_fn=[nih_five_echo_1],
            auc_fn=test_AUC,
            mask_fn=[mask_five_echo],
            output_filename="test_auc",
            tr=2.0,
            thr=0,
            out_dir=out_dir,
            te=[0],
            hrf_model="spm",
            block_model=False,
            n_jobs=1,
            use_bids=False,
            quiet=True,
            command_str="auc_to_estimates -i test.nii.gz -a auc.nii.gz -m mask.nii.gz -o out -tr 2",
        )

        # Check command file exists
        assert op.exists(op.join(out_dir, "call.sh"))
        with open(op.join(out_dir, "call.sh"), "r") as f:
            content = f.read()
        assert "auc_to_estimates" in content

    def test_auc_to_estimates_direct_threshold(
        self, nih_five_echo_1, mask_five_echo, test_AUC, testpath
    ):
        """Test auc_to_estimates with direct threshold value (0 < thr < 1)."""
        out_dir = op.join(testpath, "auc_test_direct_thr")

        auc_to_estimates.auc_to_estimates(
            data_fn=[nih_five_echo_1],
            auc_fn=test_AUC,
            mask_fn=[mask_five_echo],
            output_filename="test_auc",
            tr=2.0,
            thr=0.5,  # Direct threshold value
            out_dir=out_dir,
            te=[0],
            hrf_model="spm",
            block_model=False,
            n_jobs=1,
            use_bids=False,
            quiet=True,
        )

        # Check output files exist (non-BIDS uses _pySPFM_ prefix)
        assert op.exists(op.join(out_dir, "test_auc_pySPFM_aucThresholded.nii.gz"))
        assert op.exists(op.join(out_dir, "test_auc_pySPFM_activityInducing.nii.gz"))

    def test_auc_to_estimates_te_in_ms(self, nih_five_echo_1, mask_five_echo, test_AUC, testpath):
        """Test auc_to_estimates converts TE from ms to s."""
        out_dir = op.join(testpath, "auc_test_te_ms")

        auc_to_estimates.auc_to_estimates(
            data_fn=[nih_five_echo_1],
            auc_fn=test_AUC,
            mask_fn=[mask_five_echo],
            output_filename="test_auc",
            tr=2.0,
            thr=0,
            out_dir=out_dir,
            te=[30],  # TE in ms, should be converted to 0.030 s
            hrf_model="spm",
            block_model=False,
            n_jobs=1,
            use_bids=False,
            quiet=True,
        )

        # Check output files exist (non-BIDS uses _pySPFM_ prefix)
        assert op.exists(op.join(out_dir, "test_auc_pySPFM_aucThresholded.nii.gz"))

    def test_auc_to_estimates_percentile_without_two_masks_raises(
        self, nih_five_echo_1, mask_five_echo, test_AUC, testpath
    ):
        """Test auc_to_estimates raises error for percentile threshold with one mask."""
        out_dir = op.join(testpath, "auc_test_error")

        with pytest.raises(ValueError, match="must have two elements"):
            auc_to_estimates.auc_to_estimates(
                data_fn=[nih_five_echo_1],
                auc_fn=test_AUC,
                mask_fn=[mask_five_echo],  # Only one mask
                output_filename="test_auc",
                tr=2.0,
                thr=95,  # Percentile threshold requires two masks
                out_dir=out_dir,
                te=[0],
                hrf_model="spm",
                block_model=False,
                n_jobs=1,
                use_bids=False,
                quiet=True,
            )

    def test_auc_to_estimates_static_threshold_with_binary_mask(
        self, nih_five_echo_1, mask_five_echo, test_AUC, testpath
    ):
        """Test auc_to_estimates with static percentile threshold using binary mask."""
        out_dir = op.join(testpath, "auc_test_static_binary")

        auc_to_estimates.auc_to_estimates(
            data_fn=[nih_five_echo_1],
            auc_fn=test_AUC,
            mask_fn=[mask_five_echo, mask_five_echo],  # Binary mask for thresholding
            output_filename="test_auc",
            tr=2.0,
            thr=95,
            thr_strategy="static",
            out_dir=out_dir,
            te=[0],
            hrf_model="spm",
            block_model=False,
            n_jobs=1,
            use_bids=False,
            quiet=True,
        )

        # Check output files exist (non-BIDS uses _pySPFM_ prefix)
        assert op.exists(op.join(out_dir, "test_auc_pySPFM_aucThresholded.nii.gz"))
        assert op.exists(op.join(out_dir, "test_auc_pySPFM_activityInducing.nii.gz"))

    def test_auc_to_estimates_time_threshold_with_binary_mask(
        self, nih_five_echo_1, mask_five_echo, test_AUC, testpath
    ):
        """Test auc_to_estimates with time-dependent percentile threshold."""
        out_dir = op.join(testpath, "auc_test_time_binary")

        auc_to_estimates.auc_to_estimates(
            data_fn=[nih_five_echo_1],
            auc_fn=test_AUC,
            mask_fn=[mask_five_echo, mask_five_echo],  # Binary mask for thresholding
            output_filename="test_auc",
            tr=2.0,
            thr=95,
            thr_strategy="time",
            out_dir=out_dir,
            te=[0],
            hrf_model="spm",
            block_model=False,
            n_jobs=1,
            use_bids=False,
            quiet=True,
        )

        # Check output files exist (non-BIDS uses _pySPFM_ prefix)
        assert op.exists(op.join(out_dir, "test_auc_pySPFM_aucThresholded.nii.gz"))
        assert op.exists(op.join(out_dir, "test_auc_pySPFM_activityInducing.nii.gz"))

    def test_auc_to_estimates_3d_nonbinary_threshold(
        self, nih_five_echo_1, mask_five_echo, test_AUC, mean_AUC, testpath
    ):
        """Test auc_to_estimates with 3D non-binary threshold mask."""
        out_dir = op.join(testpath, "auc_test_3d_nonbinary")

        auc_to_estimates.auc_to_estimates(
            data_fn=[nih_five_echo_1],
            auc_fn=test_AUC,
            mask_fn=[mask_five_echo, mean_AUC],  # Non-binary 3D threshold mask
            output_filename="test_auc",
            tr=2.0,
            thr=95,  # Will be ignored since mask is not binary
            out_dir=out_dir,
            te=[0],
            hrf_model="spm",
            block_model=False,
            n_jobs=1,
            use_bids=False,
            quiet=True,
        )

        # Check output files exist (non-BIDS uses _pySPFM_ prefix)
        assert op.exists(op.join(out_dir, "test_auc_pySPFM_aucThresholded.nii.gz"))
        assert op.exists(op.join(out_dir, "test_auc_pySPFM_activityInducing.nii.gz"))

    def test_auc_to_estimates_4d_threshold(
        self, nih_five_echo_1, mask_five_echo, test_AUC, auc_4D_thr, testpath
    ):
        """Test auc_to_estimates with 4D time-dependent threshold mask."""
        out_dir = op.join(testpath, "auc_test_4d_thr")

        auc_to_estimates.auc_to_estimates(
            data_fn=[nih_five_echo_1],
            auc_fn=test_AUC,
            mask_fn=[mask_five_echo, auc_4D_thr],  # 4D threshold mask
            output_filename="test_auc",
            tr=2.0,
            thr=95,  # Will be ignored since using 4D mask
            out_dir=out_dir,
            te=[0],
            hrf_model="spm",
            block_model=False,
            n_jobs=1,
            use_bids=False,
            quiet=True,
        )

        # Check output files exist (non-BIDS uses _pySPFM_ prefix)
        assert op.exists(op.join(out_dir, "test_auc_pySPFM_aucThresholded.nii.gz"))
        assert op.exists(op.join(out_dir, "test_auc_pySPFM_activityInducing.nii.gz"))

    def test_auc_to_estimates_glover_hrf(
        self, nih_five_echo_1, mask_five_echo, test_AUC, testpath
    ):
        """Test auc_to_estimates with Glover HRF model."""
        out_dir = op.join(testpath, "auc_test_glover")

        auc_to_estimates.auc_to_estimates(
            data_fn=[nih_five_echo_1],
            auc_fn=test_AUC,
            mask_fn=[mask_five_echo],
            output_filename="test_auc",
            tr=2.0,
            thr=0,
            out_dir=out_dir,
            te=[0],
            hrf_model="glover",
            block_model=False,
            n_jobs=1,
            use_bids=False,
            quiet=True,
        )

        # Check output files exist (non-BIDS uses _pySPFM_ prefix)
        assert op.exists(op.join(out_dir, "test_auc_pySPFM_aucThresholded.nii.gz"))
        assert op.exists(op.join(out_dir, "test_auc_pySPFM_activityInducing.nii.gz"))


class TestGetKeywordDescriptionAUC:
    """Tests for get_keyword_description with AUC keywords."""

    def test_auc_keyword(self):
        """Test description for AUC keyword."""
        from pySPFM import utils

        desc = utils.get_keyword_description("AUC")
        assert "area under the curve" in desc.lower()

    def test_auc_thresholded_keyword(self):
        """Test description for aucThresholded keyword."""
        from pySPFM import utils

        desc = utils.get_keyword_description("aucThresholded")
        assert "area under the curve" in desc.lower()

    def test_unknown_keyword_fallback(self):
        """Test description for unknown keyword returns fallback."""
        from pySPFM import utils

        desc = utils.get_keyword_description("unknownKeyword123")
        assert "pySPFM output" in desc
        assert "unknownKeyword123" in desc
