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

# Video and rendering related (including Vulkan for Isaac Sim GPU plugin)
apt-get install -y libgl1 libglib2.0-0 libxt6 libglu1-mesa ffmpeg \
    libvulkan1 libegl1 libxrandr2 libxrender1 libxext6 libxkbcommon0 \
    xvfb x11-utils

# Use host X11 display if reachable; otherwise start a virtual display (Xvfb)
# so Isaac Sim can initialize its rendering subsystem even without a physical display.
if [ -z "${DISPLAY:-}" ] || ! xdpyinfo -display "${DISPLAY}" >/dev/null 2>&1; then
    echo "Host display not available — starting virtual display (Xvfb :99)"
    Xvfb :99 -screen 0 1920x1080x24 &
    export DISPLAY=:99
fi

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Create venv with Python 3.11 and install the project
rm -rf /workspace/.venv
uv venv --python python3.11 /workspace/.venv
source /workspace/.venv/bin/activate
cd /workspace
uv sync

# Mark /workspace as safe so git works correctly when running as root
# (required since git 2.35.2 — avoids "dubious ownership" error on bind-mounted dirs) so that ClearML can properly report repo and commit
git config --global --add safe.directory /workspace

# Automatically accept Isaac Sim/Lab EULA
export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=Y

# Run the provided command, or drop into an interactive shell
exec "${@:-bash}"
