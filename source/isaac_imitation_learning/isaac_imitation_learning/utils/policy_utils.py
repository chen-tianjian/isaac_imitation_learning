# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility functions for robomimic policy inference."""

import numpy as np

ACTION_CHUNKING_POLICY_LIST = ["diffusion_policy", "act"]


def is_action_chunking_policy(policy) -> bool:
    """Check if the policy is an action chunking policy that requires observation stacking.

    Args:
        policy: The robomimic RolloutPolicy.

    Returns:
        bool: True if the policy is an action chunking policy.
    """
    algo_name = policy.policy.global_config.algo_name
    return algo_name in ACTION_CHUNKING_POLICY_LIST


def get_observation_horizon(policy) -> int:
    """Get the observation horizon (frame stack size) for an action chunking policy.

    Args:
        policy: The robomimic RolloutPolicy.

    Returns:
        int: The observation horizon (frame stack size).
    """
    return policy.policy.algo_config.horizon.observation_horizon


def stack_observations(obs_queue, obs_keys) -> dict:
    """Stack observations from queue along time dimension.

    Args:
        obs_queue: Deque of observation dictionaries.
        obs_keys: List of observation keys.

    Returns:
        dict: Stacked observations with shape [T, ...] for each key.
    """
    stacked_obs = {}
    for key in obs_keys:
        # Stack observations along time dimension (axis 0)
        obs_list = [obs[key] for obs in obs_queue]
        stacked_obs[key] = np.stack(obs_list, axis=0)
    return stacked_obs
