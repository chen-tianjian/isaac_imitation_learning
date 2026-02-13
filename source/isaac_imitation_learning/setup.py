# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'isaac_imitation_learning' python package."""

from setuptools import find_packages, setup

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "psutil",
    "clearml",
    "robomimic @ git+https://github.com/chen-tianjian/robomimic.git@act",
]

# Installation operation
setup(
    name="isaac_imitation_learning",
    packages=find_packages(),
    author="Tianjian Chen",
    maintainer="Tianjian Chen",
    url="https://github.com/chen-tianjian/isaac_imitation_learning",
    version="0.1.0",
    description="Imitation learning with Isaac Lab/Sim and Robomimic (ACT, Diffusion Policy, BC)",
    keywords=["imitation-learning", "robomimic", "isaaclab", "act", "diffusion-policy"],
    install_requires=INSTALL_REQUIRES,
    license="Apache-2.0",
    include_package_data=True,
    package_data={
        "isaac_imitation_learning": ["tasks/**/agents/robomimic/*.json"],
    },
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Isaac Sim :: 4.5.0",
        "Isaac Sim :: 5.0.0",
        "Isaac Sim :: 5.1.0",
    ],
    zip_safe=False,
)
