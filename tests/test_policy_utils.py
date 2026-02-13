# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for policy_utils.py."""

import unittest
from unittest.mock import MagicMock

import numpy as np


class TestIsActionChunkingPolicy(unittest.TestCase):
    """Test is_action_chunking_policy function."""

    def test_act_policy_returns_true(self):
        from isaac_imitation_learning.utils.policy_utils import is_action_chunking_policy

        policy = MagicMock()
        policy.policy.global_config.algo_name = "act"
        self.assertTrue(is_action_chunking_policy(policy))

    def test_diffusion_policy_returns_true(self):
        from isaac_imitation_learning.utils.policy_utils import is_action_chunking_policy

        policy = MagicMock()
        policy.policy.global_config.algo_name = "diffusion_policy"
        self.assertTrue(is_action_chunking_policy(policy))

    def test_bc_rnn_returns_false(self):
        from isaac_imitation_learning.utils.policy_utils import is_action_chunking_policy

        policy = MagicMock()
        policy.policy.global_config.algo_name = "bc"
        self.assertFalse(is_action_chunking_policy(policy))

    def test_unknown_algo_returns_false(self):
        from isaac_imitation_learning.utils.policy_utils import is_action_chunking_policy

        policy = MagicMock()
        policy.policy.global_config.algo_name = "some_other_algo"
        self.assertFalse(is_action_chunking_policy(policy))


class TestGetObservationHorizon(unittest.TestCase):
    """Test get_observation_horizon function."""

    def test_returns_observation_horizon(self):
        from isaac_imitation_learning.utils.policy_utils import get_observation_horizon

        policy = MagicMock()
        policy.policy.algo_config.horizon.observation_horizon = 4
        self.assertEqual(get_observation_horizon(policy), 4)

    def test_returns_one(self):
        from isaac_imitation_learning.utils.policy_utils import get_observation_horizon

        policy = MagicMock()
        policy.policy.algo_config.horizon.observation_horizon = 1
        self.assertEqual(get_observation_horizon(policy), 1)


class TestStackObservations(unittest.TestCase):
    """Test stack_observations function."""

    def test_stacks_along_time_dimension(self):
        from isaac_imitation_learning.utils.policy_utils import stack_observations

        obs1 = {"pos": np.array([1.0, 2.0, 3.0]), "vel": np.array([0.1, 0.2])}
        obs2 = {"pos": np.array([4.0, 5.0, 6.0]), "vel": np.array([0.3, 0.4])}

        result = stack_observations([obs1, obs2], ["pos", "vel"])

        np.testing.assert_array_equal(result["pos"], np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        np.testing.assert_array_equal(result["vel"], np.array([[0.1, 0.2], [0.3, 0.4]]))

    def test_single_observation(self):
        from isaac_imitation_learning.utils.policy_utils import stack_observations

        obs = {"pos": np.array([1.0, 2.0])}
        result = stack_observations([obs], ["pos"])

        np.testing.assert_array_equal(result["pos"], np.array([[1.0, 2.0]]))

    def test_image_observations(self):
        from isaac_imitation_learning.utils.policy_utils import stack_observations

        img1 = {"image": np.ones((64, 64, 3))}
        img2 = {"image": np.zeros((64, 64, 3))}

        result = stack_observations([img1, img2], ["image"])

        self.assertEqual(result["image"].shape, (2, 64, 64, 3))


if __name__ == "__main__":
    unittest.main()
