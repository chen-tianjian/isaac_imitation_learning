# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Augment Stack-Cube environment gym specs with ACT and Diffusion Policy entry points.

The base environments (Isaac-Stack-Cube-Franka-IK-Rel-v0 and
Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0) are already registered by
pip-installed isaaclab_tasks with BC-RNN entry points. This module adds
the ACT and Diffusion Policy entry points to those existing specs.
"""

import gymnasium as gym

from . import agents


def _augment_spec(env_id: str, diffusion_json: str, act_json: str) -> None:
    """Augment a gym spec with Diffusion Policy and ACT entry points."""
    spec = gym.spec(env_id)
    spec.kwargs["robomimic_diffusion_policy_cfg_entry_point"] = f"{agents.__name__}:{diffusion_json}"
    spec.kwargs["robomimic_act_cfg_entry_point"] = f"{agents.__name__}:{act_json}"


##
# Augment existing Gym environment specs.
##

_augment_spec(
    "Isaac-Stack-Cube-Franka-IK-Rel-v0",
    "robomimic/diffusion_policy_low_dim.json",
    "robomimic/act_low_dim.json",
)

_augment_spec(
    "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0",
    "robomimic/diffusion_policy_image.json",
    "robomimic/act_image.json",
)
