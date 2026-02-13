# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for the extension."""

##
# Register Gym environments.
##

try:
    from isaaclab_tasks.utils import import_packages

    # The blacklist is used to prevent importing configs from sub-packages
    _BLACKLIST_PKGS = ["utils", ".mdp", "agents"]
    # Import all configs in this package
    import_packages(__name__, _BLACKLIST_PKGS)
except ImportError:
    pass  # isaaclab_tasks not installed; env spec augmentation will be skipped
