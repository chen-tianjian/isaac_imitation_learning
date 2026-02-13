# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for clearml_utils.py.

These tests run without Isaac Sim and without ClearML installed.
They use unittest.mock to simulate ClearML SDK interactions.
"""

import argparse
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch


class TestClearMLURIParsing(unittest.TestCase):
    """Test URI parsing and detection functions."""

    def test_is_clearml_uri_with_valid_uri(self):
        from isaac_imitation_learning.utils.clearml_utils import is_clearml_uri

        self.assertTrue(is_clearml_uri("clearml://abc123"))
        self.assertTrue(is_clearml_uri("clearml://abc123/file.hdf5"))
        self.assertTrue(is_clearml_uri("clearml://abc123def456/subdir/file.pth"))

    def test_is_clearml_uri_with_local_path(self):
        from isaac_imitation_learning.utils.clearml_utils import is_clearml_uri

        self.assertFalse(is_clearml_uri("/path/to/file.hdf5"))
        self.assertFalse(is_clearml_uri("relative/path/file.pth"))
        self.assertFalse(is_clearml_uri("./file.hdf5"))
        self.assertFalse(is_clearml_uri(""))

    def test_is_clearml_uri_with_none(self):
        from isaac_imitation_learning.utils.clearml_utils import is_clearml_uri

        self.assertFalse(is_clearml_uri(None))

    def test_parse_clearml_uri_id_only(self):
        from isaac_imitation_learning.utils.clearml_utils import parse_clearml_uri

        resource_id, filename = parse_clearml_uri("clearml://abc123def456")
        self.assertEqual(resource_id, "abc123def456")
        self.assertIsNone(filename)

    def test_parse_clearml_uri_with_filename(self):
        from isaac_imitation_learning.utils.clearml_utils import parse_clearml_uri

        resource_id, filename = parse_clearml_uri("clearml://abc123/model_epoch_100.pth")
        self.assertEqual(resource_id, "abc123")
        self.assertEqual(filename, "model_epoch_100.pth")

        # Test with deeper path
        resource_id, filename = parse_clearml_uri("clearml://abc123/subdir/file.hdf5")
        self.assertEqual(resource_id, "abc123")
        self.assertEqual(filename, "subdir/file.hdf5")

    def test_parse_clearml_uri_malformed_raises(self):
        from isaac_imitation_learning.utils.clearml_utils import parse_clearml_uri

        with self.assertRaises(ValueError):
            parse_clearml_uri("not_a_clearml_uri")

        with self.assertRaises(ValueError):
            parse_clearml_uri("clearml://")

        with self.assertRaises(ValueError):
            parse_clearml_uri("")


class TestGracefulDegradation(unittest.TestCase):
    """Test that all functions gracefully no-op when task is None or ClearML unavailable."""

    def test_is_clearml_configured_returns_bool(self):
        from isaac_imitation_learning.utils.clearml_utils import is_clearml_configured

        # The function should return a boolean without raising
        result = is_clearml_configured()
        self.assertIsInstance(result, bool)

    def test_init_clearml_task_returns_none_when_unavailable(self):
        from isaac_imitation_learning.utils.clearml_utils import init_clearml_task

        args = argparse.Namespace(no_clearml=True, clearml_project="test", clearml_task_name=None)
        result = init_clearml_task(args, task_type="training", default_task_name="test_task")
        self.assertIsNone(result)

    def test_upload_checkpoint_noop_when_task_none(self):
        from isaac_imitation_learning.utils.clearml_utils import upload_checkpoint

        # Should not raise
        upload_checkpoint(None, "/path/to/model.pth")

    def test_connect_configuration_dict_noop_when_task_none(self):
        from isaac_imitation_learning.utils.clearml_utils import connect_configuration_dict

        # Should not raise
        connect_configuration_dict(None, {"key": "value"}, "test_config")

    def test_connect_configuration_file_noop_when_task_none(self):
        from isaac_imitation_learning.utils.clearml_utils import connect_configuration_file

        # Should not raise
        connect_configuration_file(None, "/path/to/config.json", "test_config")

    def test_report_scalar_noop_when_task_none(self):
        from isaac_imitation_learning.utils.clearml_utils import report_scalar

        # Should not raise
        report_scalar(None, "title", "series", 0.95, 1)

    def test_report_video_noop_when_task_none(self):
        from isaac_imitation_learning.utils.clearml_utils import report_video

        # Should not raise
        report_video(None, "/path/to/video.mp4", "title", "series", 1)

    def test_upload_videos_from_dir_noop_when_task_none(self):
        from isaac_imitation_learning.utils.clearml_utils import upload_videos_from_dir

        # Should not raise
        upload_videos_from_dir(None, "/path/to/videos", "title")

    def test_close_clearml_task_noop_when_task_none(self):
        from isaac_imitation_learning.utils.clearml_utils import close_clearml_task

        # Should not raise
        close_clearml_task(None)


class TestCloseClearMLTask(unittest.TestCase):
    """Test close_clearml_task with mocked ClearML."""

    def test_close_calls_task_close(self):
        from isaac_imitation_learning.utils.clearml_utils import close_clearml_task

        mock_task = MagicMock()
        close_clearml_task(mock_task)

        mock_task.close.assert_called_once()

    def test_noop_when_task_none(self):
        from isaac_imitation_learning.utils.clearml_utils import close_clearml_task

        # Should not raise
        close_clearml_task(None)


class TestAddClearMLArgs(unittest.TestCase):
    """Test that add_clearml_args adds all expected arguments."""

    def setUp(self):
        self.parser = argparse.ArgumentParser()

    def test_adds_all_expected_args(self):
        from isaac_imitation_learning.utils.clearml_utils import add_clearml_args

        add_clearml_args(self.parser)

        # Parse with no args to check defaults exist
        args = self.parser.parse_args([])
        self.assertTrue(hasattr(args, "clearml_project"))
        self.assertTrue(hasattr(args, "clearml_task_name"))
        self.assertTrue(hasattr(args, "clearml_queue"))
        self.assertTrue(hasattr(args, "remote"))
        self.assertTrue(hasattr(args, "no_clearml"))

    def test_default_values(self):
        from isaac_imitation_learning.utils.clearml_utils import add_clearml_args

        add_clearml_args(self.parser)
        args = self.parser.parse_args([])

        self.assertEqual(args.clearml_project, "IsaacLab/Robomimic")
        self.assertIsNone(args.clearml_task_name)
        self.assertEqual(args.clearml_queue, "default")
        self.assertFalse(args.remote)
        self.assertFalse(args.no_clearml)

    def test_remote_flag(self):
        from isaac_imitation_learning.utils.clearml_utils import add_clearml_args

        add_clearml_args(self.parser)
        args = self.parser.parse_args(["--remote"])
        self.assertTrue(args.remote)

    def test_no_clearml_flag(self):
        from isaac_imitation_learning.utils.clearml_utils import add_clearml_args

        add_clearml_args(self.parser)
        args = self.parser.parse_args(["--no_clearml"])
        self.assertTrue(args.no_clearml)


class TestResolveDataset(unittest.TestCase):
    """Test dataset resolution with mocked ClearML."""

    def test_local_path_passthrough(self):
        from isaac_imitation_learning.utils.clearml_utils import resolve_dataset

        result = resolve_dataset("/path/to/data.hdf5")
        self.assertEqual(result, "/path/to/data.hdf5")

    def test_clearml_uri_single_hdf5(self):
        from isaac_imitation_learning.utils.clearml_utils import resolve_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a single HDF5 file in the temp dir
            hdf5_path = os.path.join(tmpdir, "data.hdf5")
            with open(hdf5_path, "w") as f:
                f.write("fake hdf5")

            mock_dataset = MagicMock()
            mock_dataset.get_local_copy.return_value = tmpdir

            with patch("isaac_imitation_learning.utils.clearml_utils.Dataset") as MockDataset:
                MockDataset.get.return_value = mock_dataset
                result = resolve_dataset("clearml://dataset123")

            self.assertEqual(result, hdf5_path)

    def test_clearml_uri_multiple_hdf5_raises_without_filename(self):
        from isaac_imitation_learning.utils.clearml_utils import resolve_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple HDF5 files
            for name in ["data1.hdf5", "data2.hdf5"]:
                with open(os.path.join(tmpdir, name), "w") as f:
                    f.write("fake hdf5")

            mock_dataset = MagicMock()
            mock_dataset.get_local_copy.return_value = tmpdir

            with patch("isaac_imitation_learning.utils.clearml_utils.Dataset") as MockDataset:
                MockDataset.get.return_value = mock_dataset
                with self.assertRaises(ValueError):
                    resolve_dataset("clearml://dataset123")

    def test_clearml_uri_with_filename(self):
        from isaac_imitation_learning.utils.clearml_utils import resolve_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple HDF5 files
            for name in ["data1.hdf5", "data2.hdf5"]:
                with open(os.path.join(tmpdir, name), "w") as f:
                    f.write("fake hdf5")

            mock_dataset = MagicMock()
            mock_dataset.get_local_copy.return_value = tmpdir

            with patch("isaac_imitation_learning.utils.clearml_utils.Dataset") as MockDataset:
                MockDataset.get.return_value = mock_dataset
                result = resolve_dataset("clearml://dataset123/data1.hdf5")

            self.assertEqual(result, os.path.join(tmpdir, "data1.hdf5"))


class TestResolveCheckpoint(unittest.TestCase):
    """Test checkpoint resolution with mocked ClearML."""

    def test_local_path_passthrough(self):
        from isaac_imitation_learning.utils.clearml_utils import resolve_checkpoint

        result = resolve_checkpoint("/path/to/model.pth")
        self.assertEqual(result, "/path/to/model.pth")

    def test_clearml_uri_latest_checkpoint(self):
        from isaac_imitation_learning.utils.clearml_utils import resolve_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create checkpoint files
            for name in ["model_epoch_50.pth", "model_epoch_100.pth", "model_epoch_150.pth"]:
                with open(os.path.join(tmpdir, name), "w") as f:
                    f.write("fake checkpoint")

            mock_task = MagicMock()
            # Simulate artifacts dict
            mock_task.artifacts = {
                "model_epoch_50.pth": MagicMock(),
                "model_epoch_100.pth": MagicMock(),
                "model_epoch_150.pth": MagicMock(),
            }
            for name, mock_art in mock_task.artifacts.items():
                mock_art.get_local_copy.return_value = os.path.join(tmpdir, name)

            with patch("isaac_imitation_learning.utils.clearml_utils.Task") as MockTask:
                MockTask.get_task.return_value = mock_task
                result = resolve_checkpoint("clearml://task123")

            # Should return the latest checkpoint (epoch 150)
            self.assertEqual(result, os.path.join(tmpdir, "model_epoch_150.pth"))

    def test_clearml_uri_specific_checkpoint(self):
        from isaac_imitation_learning.utils.clearml_utils import resolve_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "model_epoch_100.pth")
            with open(ckpt_path, "w") as f:
                f.write("fake checkpoint")

            mock_task = MagicMock()
            mock_artifact = MagicMock()
            mock_artifact.get_local_copy.return_value = ckpt_path
            mock_task.artifacts = {"model_epoch_100.pth": mock_artifact}

            with patch("isaac_imitation_learning.utils.clearml_utils.Task") as MockTask:
                MockTask.get_task.return_value = mock_task
                result = resolve_checkpoint("clearml://task123/model_epoch_100.pth")

            self.assertEqual(result, ckpt_path)


class TestResolveInputDir(unittest.TestCase):
    """Test input_dir resolution with mocked ClearML."""

    def test_local_path_passthrough(self):
        from isaac_imitation_learning.utils.clearml_utils import resolve_input_dir

        result = resolve_input_dir("/path/to/checkpoints/")
        self.assertEqual(result, "/path/to/checkpoints/")

    def test_clearml_uri_downloads_all_checkpoints(self):
        from isaac_imitation_learning.utils.clearml_utils import resolve_input_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create checkpoint files
            ckpt_names = ["model_epoch_50.pth", "model_epoch_100.pth"]
            for name in ckpt_names:
                with open(os.path.join(tmpdir, name), "w") as f:
                    f.write("fake checkpoint")

            mock_task = MagicMock()
            mock_task.artifacts = {}
            for name in ckpt_names:
                mock_art = MagicMock()
                mock_art.get_local_copy.return_value = os.path.join(tmpdir, name)
                mock_task.artifacts[name] = mock_art

            with patch("isaac_imitation_learning.utils.clearml_utils.Task") as MockTask:
                MockTask.get_task.return_value = mock_task
                result_dir = resolve_input_dir("clearml://task123")

            # Result should be a directory containing the downloaded checkpoints
            self.assertTrue(os.path.isdir(result_dir))
            downloaded_files = os.listdir(result_dir)
            for name in ckpt_names:
                self.assertIn(name, downloaded_files)


class TestUploadCheckpoint(unittest.TestCase):
    """Test checkpoint upload with mocked ClearML."""

    def test_uploads_as_artifact_with_filename(self):
        from isaac_imitation_learning.utils.clearml_utils import upload_checkpoint

        mock_task = MagicMock()
        upload_checkpoint(mock_task, "/path/to/model_epoch_100.pth")

        mock_task.upload_artifact.assert_called_once_with(
            name="model_epoch_100.pth",
            artifact_object="/path/to/model_epoch_100.pth",
        )

    def test_noop_when_task_none(self):
        from isaac_imitation_learning.utils.clearml_utils import upload_checkpoint

        # Should not raise
        upload_checkpoint(None, "/path/to/model.pth")


class TestConnectConfiguration(unittest.TestCase):
    """Test connect_configuration functions with mocked ClearML."""

    def test_connect_dict_calls_task_connect_configuration(self):
        from isaac_imitation_learning.utils.clearml_utils import connect_configuration_dict

        mock_task = MagicMock()
        config = {"algo": "act", "epochs": 100}
        connect_configuration_dict(mock_task, config, "robomimic_config")

        mock_task.connect_configuration.assert_called_once_with(config, name="robomimic_config")

    def test_connect_file_calls_task_connect_configuration(self):
        from isaac_imitation_learning.utils.clearml_utils import connect_configuration_file

        mock_task = MagicMock()
        connect_configuration_file(mock_task, "/path/to/config.json", "robomimic_config_file")

        mock_task.connect_configuration.assert_called_once_with("/path/to/config.json", name="robomimic_config_file")

    def test_connect_dict_noop_when_task_none(self):
        from isaac_imitation_learning.utils.clearml_utils import connect_configuration_dict

        # Should not raise
        connect_configuration_dict(None, {"key": "value"}, "test")


class TestReportScalar(unittest.TestCase):
    """Test scalar reporting with mocked ClearML."""

    def test_reports_to_logger(self):
        from isaac_imitation_learning.utils.clearml_utils import report_scalar

        mock_task = MagicMock()
        mock_logger = MagicMock()
        mock_task.get_logger.return_value = mock_logger

        report_scalar(mock_task, "evaluation", "success_rate", 0.85, 1)

        mock_logger.report_scalar.assert_called_once_with(
            title="evaluation", series="success_rate", value=0.85, iteration=1
        )

    def test_noop_when_task_none(self):
        from isaac_imitation_learning.utils.clearml_utils import report_scalar

        # Should not raise
        report_scalar(None, "title", "series", 0.5, 1)


class TestReportVideo(unittest.TestCase):
    """Test video reporting with mocked ClearML."""

    def test_report_media_called(self):
        from isaac_imitation_learning.utils.clearml_utils import report_video

        mock_task = MagicMock()
        mock_logger = MagicMock()
        mock_task.get_logger.return_value = mock_logger

        report_video(mock_task, "/path/to/video.mp4", "rollouts", "episode_1", 0)

        mock_logger.report_media.assert_called_once_with(
            title="rollouts",
            series="episode_1",
            local_path="/path/to/video.mp4",
            iteration=0,
        )

    def test_noop_when_task_none(self):
        from isaac_imitation_learning.utils.clearml_utils import report_video

        # Should not raise
        report_video(None, "/path/to/video.mp4", "title", "series", 0)


class TestUploadVideosFromDir(unittest.TestCase):
    """Test batch video upload from directory with mocked ClearML."""

    def test_finds_and_reports_mp4_files(self):
        from isaac_imitation_learning.utils.clearml_utils import upload_videos_from_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some mp4 files
            for name in ["video_0.mp4", "video_1.mp4"]:
                with open(os.path.join(tmpdir, name), "w") as f:
                    f.write("fake video")
            # Create a non-mp4 file (should be ignored)
            with open(os.path.join(tmpdir, "notes.txt"), "w") as f:
                f.write("not a video")

            mock_task = MagicMock()
            mock_logger = MagicMock()
            mock_task.get_logger.return_value = mock_logger

            upload_videos_from_dir(mock_task, tmpdir, "training_rollouts")

            # Should have called report_media twice (once per mp4)
            self.assertEqual(mock_logger.report_media.call_count, 2)

    def test_noop_on_empty_dir(self):
        from isaac_imitation_learning.utils.clearml_utils import upload_videos_from_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_task = MagicMock()
            mock_logger = MagicMock()
            mock_task.get_logger.return_value = mock_logger

            upload_videos_from_dir(mock_task, tmpdir, "training_rollouts")

            mock_logger.report_media.assert_not_called()

    def test_noop_when_task_none(self):
        from isaac_imitation_learning.utils.clearml_utils import upload_videos_from_dir

        # Should not raise
        upload_videos_from_dir(None, "/path/to/videos", "title")


if __name__ == "__main__":
    unittest.main()
