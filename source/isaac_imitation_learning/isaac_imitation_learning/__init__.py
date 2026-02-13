# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Imitation Learning extension package.

Augments pip-installed isaaclab_tasks environments with additional
robomimic policy entry points (ACT, Diffusion Policy).

Note:
    Env spec augmentation is NOT triggered on ``import isaac_imitation_learning``
    because utility imports (e.g. clearml_utils) happen before AppLauncher, when
    isaaclab_tasks envs are not yet registered. Scripts must explicitly call::

        import isaac_imitation_learning.tasks

    after the AppLauncher has started.
"""
