#!/bin/bash
# Docker init script for Isaac Imitation Learning.
# Installs Python 3.11, uv, creates a venv, and installs the project.
# Used as the entrypoint in docker-compose.yml for local development.
set -euo pipefail

# Set timezone
export DEBIAN_FRONTEND=noninteractive
ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime

# System deps + Python 3.11 (Ubuntu 24.04 ships 3.12; Isaac Sim/Lab needs 3.11)
apt-get update && apt-get install -y software-properties-common git curl tzdata cmake build-essential
dpkg-reconfigure --frontend noninteractive tzdata
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update && apt-get install -y python3.11 python3.11-dev python3.11-venv

# Video related
apt-get install -y libgl1 libglib2.0-0 libxt6 libglu1-mesa ffmpeg

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Create venv with Python 3.11 and install the project
rm -rf /workspace/.venv
uv venv --python python3.11 /workspace/.venv
source /workspace/.venv/bin/activate
cd /workspace
uv sync

# Automatically accept Isaac Sim/Lab EULA
export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=Y

# Run the provided command, or drop into an interactive shell
exec "${@:-bash}"
