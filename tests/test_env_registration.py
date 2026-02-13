# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for environment registration augmentation.

These tests verify that importing isaac_imitation_learning augments the
gym specs for Stack-Cube environments with ACT and Diffusion Policy entry points
while preserving the existing BC entry points from isaaclab_tasks.

NOTE: These tests require isaaclab_tasks to be installed (pip install isaaclab).
They will be skipped if isaaclab_tasks is not available.
"""

import importlib
import os
import unittest

try:
    import isaaclab_tasks  # noqa: F401

    HAS_ISAACLAB_TASKS = True
except ImportError:
    HAS_ISAACLAB_TASKS = False


@unittest.skipUnless(HAS_ISAACLAB_TASKS, "isaaclab_tasks not installed")
class TestStackCubeIKRelRegistration(unittest.TestCase):
    """Test that Isaac-Stack-Cube-Franka-IK-Rel-v0 has all expected entry points."""

    @classmethod
    def setUpClass(cls):
        import gymnasium as gym

        import isaac_imitation_learning  # noqa: F401 - triggers spec augmentation

        cls.spec = gym.spec("Isaac-Stack-Cube-Franka-IK-Rel-v0")

    def test_has_bc_entry_point(self):
        """Original BC entry point from isaaclab_tasks should be preserved."""
        self.assertIn("robomimic_bc_cfg_entry_point", self.spec.kwargs)

    def test_has_act_entry_point(self):
        """ACT entry point should be added by isaac_imitation_learning."""
        self.assertIn("robomimic_act_cfg_entry_point", self.spec.kwargs)

    def test_has_diffusion_policy_entry_point(self):
        """Diffusion Policy entry point should be added by isaac_imitation_learning."""
        self.assertIn("robomimic_diffusion_policy_cfg_entry_point", self.spec.kwargs)

    def test_act_config_resolves_to_file(self):
        """ACT entry point should resolve to an actual JSON file."""
        entry_point = self.spec.kwargs["robomimic_act_cfg_entry_point"]
        mod_name, file_name = entry_point.split(":")
        mod = importlib.import_module(mod_name)
        config_path = os.path.join(os.path.dirname(mod.__file__), file_name)
        self.assertTrue(os.path.exists(config_path), f"Config file not found: {config_path}")

    def test_diffusion_policy_config_resolves_to_file(self):
        """Diffusion Policy entry point should resolve to an actual JSON file."""
        entry_point = self.spec.kwargs["robomimic_diffusion_policy_cfg_entry_point"]
        mod_name, file_name = entry_point.split(":")
        mod = importlib.import_module(mod_name)
        config_path = os.path.join(os.path.dirname(mod.__file__), file_name)
        self.assertTrue(os.path.exists(config_path), f"Config file not found: {config_path}")


@unittest.skipUnless(HAS_ISAACLAB_TASKS, "isaaclab_tasks not installed")
class TestStackCubeIKRelVisuomotorRegistration(unittest.TestCase):
    """Test that Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0 has all expected entry points."""

    @classmethod
    def setUpClass(cls):
        import gymnasium as gym

        import isaac_imitation_learning  # noqa: F401 - triggers spec augmentation

        cls.spec = gym.spec("Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0")

    def test_has_bc_entry_point(self):
        """Original BC entry point from isaaclab_tasks should be preserved."""
        self.assertIn("robomimic_bc_cfg_entry_point", self.spec.kwargs)

    def test_has_act_entry_point(self):
        """ACT entry point should be added by isaac_imitation_learning."""
        self.assertIn("robomimic_act_cfg_entry_point", self.spec.kwargs)

    def test_has_diffusion_policy_entry_point(self):
        """Diffusion Policy entry point should be added by isaac_imitation_learning."""
        self.assertIn("robomimic_diffusion_policy_cfg_entry_point", self.spec.kwargs)

    def test_act_config_resolves_to_file(self):
        """ACT entry point should resolve to an actual JSON file."""
        entry_point = self.spec.kwargs["robomimic_act_cfg_entry_point"]
        mod_name, file_name = entry_point.split(":")
        mod = importlib.import_module(mod_name)
        config_path = os.path.join(os.path.dirname(mod.__file__), file_name)
        self.assertTrue(os.path.exists(config_path), f"Config file not found: {config_path}")

    def test_diffusion_policy_config_resolves_to_file(self):
        """Diffusion Policy entry point should resolve to an actual JSON file."""
        entry_point = self.spec.kwargs["robomimic_diffusion_policy_cfg_entry_point"]
        mod_name, file_name = entry_point.split(":")
        mod = importlib.import_module(mod_name)
        config_path = os.path.join(os.path.dirname(mod.__file__), file_name)
        self.assertTrue(os.path.exists(config_path), f"Config file not found: {config_path}")


if __name__ == "__main__":
    unittest.main()
