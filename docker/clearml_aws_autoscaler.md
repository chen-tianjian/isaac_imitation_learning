# ClearML AWS Autoscaler Configuration

Configuration for running Isaac Imitation Learning training jobs on ClearML AWS autoscaler
with Docker mode enabled.

## Prerequisites

- AWS AMI with NVIDIA drivers and NVIDIA Container Toolkit pre-installed
- Docker mode enabled in the autoscaler

## Autoscaler Settings

### AMI
```
ami-01f3b8bbe6f7238ca
```
(Ubuntu 24.04 with NVIDIA Driver)

### Base Docker Image

```
nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
```

### Init Script

Runs on the EC2 host after boot. Minimal since the AMI already has NVIDIA drivers
and Container Toolkit.

```bash
#!/bin/bash
systemctl start docker
```

### Additional ClearML Configuration

Appended to the agent's `clearml.conf`. Installs Python 3.11 and uv inside the
Docker container via `extra_docker_shell_script`, then uses uv as the package manager.

When `package_manager.type: uv` is set and the task has a git repository, the agent
runs `uv sync` from the repo's `pyproject.toml` (which contains all index URLs and
dependencies) instead of using pip with captured requirements.

```
agent {
    extra_docker_arguments: ["--ipc=host", "--gpus=all", "-e", "NVIDIA_DRIVER_CAPABILITIES=all"]
    package_manager {
        type: uv
        extra_index_url: ["https://pypi.nvidia.com"]
    }
    python_binary: "/usr/bin/python3.11"
    extra_docker_shell_script: [
        "nvidia-smi || echo 'DIAGNOSTIC: nvidia-smi FAILED - GPU not visible in container'",
        "echo \"DIAGNOSTIC: NVIDIA_DRIVER_CAPABILITIES=$NVIDIA_DRIVER_CAPABILITIES\"",
        "ls /usr/lib/x86_64-linux-gnu/libGLX_nvidia* 2>/dev/null || echo 'DIAGNOSTIC: libGLX_nvidia not found'",
        "export DEBIAN_FRONTEND=noninteractive",
        "ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime",
        "apt-get update",
        "apt-get install -y software-properties-common git curl tzdata cmake build-essential",
        "dpkg-reconfigure --frontend noninteractive tzdata",
        "add-apt-repository -y ppa:deadsnakes/ppa",
        "apt-get update",
        "apt-get install -y python3.11 python3.11-dev python3.11-venv",
        "apt-get install -y libgl1 libglib2.0-0 libxt6 libglu1-mesa ffmpeg libvulkan1 libegl1 libxrandr2 libxrender1 libxext6 libxkbcommon0 xvfb x11-utils",
        "mkdir -p /usr/share/vulkan/icd.d",
        "printf '{\"file_format_version\":\"1.0.0\",\"ICD\":{\"library_path\":\"libGLX_nvidia.so.0\",\"api_version\":\"1.3\"}}' > /usr/share/vulkan/icd.d/nvidia_icd.json",
        "(Xvfb :99 -screen 0 1920x1080x24 &)",
        "export DISPLAY=:99",
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "export PATH=$HOME/.local/bin:$PATH",
        "export ACCEPT_EULA=Y",
        "export OMNI_KIT_ACCEPT_EULA=Y"
    ]
    docker_force_pull: true
}
```

### What the agent does

1. Pulls the stock NVIDIA CUDA image
2. Runs `extra_docker_shell_script` inside the container (installs Python 3.11 + uv)
3. Clones the git repository
4. Runs `uv sync` from `pyproject.toml` (installs all dependencies with correct index URLs)
5. Runs the task headless

## Submitting a Remote Job

```bash
python scripts/robomimic/train.py \
    --task=Isaac-Stack-Cube-Franka-IK-Rel-v0 \
    --algo=act \
    --dataset=clearml://<DATASET_ID>/data.hdf5 \
    --remote
```

The `--remote` flag sends the task to the ClearML queue and exits locally.
The autoscaler picks it up and runs it on a GPU instance.

## Troubleshooting

### `isaaclab` / `isaacsim` not found during pip install

The agent fell back to pip instead of `uv sync`. Check:

1. **Task has empty `repository` field** — the agent can't clone code or find
   `pyproject.toml`, so `uv sync` never runs. Verify that `Task.init()` captured
   the git repository by checking the task's "Execution" tab in the ClearML UI.
   The script must be run from inside the git working tree with a valid remote.
2. **`extra_index_url`** — as a safety net, `https://pypi.nvidia.com` is configured
   above so pip can find NVIDIA packages even in fallback mode.

### `extra_docker_shell_script` not running / cmake not found during `uv sync`

**Symptom:** Task log shows `FileNotFoundError: cmake` when building `egl-probe`, and
the apt output for cmake never appears before `uv sync` starts.

**Root cause:** clearml-agent reads `extra_docker_shell_script` from config key
`agent.extra_docker_shell_script`. If the setting is nested inside `agent.package_manager`
instead of at the `agent` level, the agent silently ignores it and the script never runs.

**Diagnosis:** In the task log, search for `agent.extra_docker_shell_script` in the config
dump section. If you instead see `agent.package_manager.extra_docker_shell_script`, the
nesting is wrong.

**Fix:** In the autoscaler "Additional ClearML Configuration", make sure `python_binary`,
`extra_docker_shell_script`, and `docker_force_pull` are directly under `agent {}` — not
inside `agent.package_manager {}`. Compare against the template above.

**Mitigation already in pyproject.toml:** `cmake` is a project dependency (provides the
cmake binary via PyPI) and `egl-probe` uses `no-build-isolation` so it picks up that
binary. This means `egl-probe` builds successfully even when the shell script does not run.

### GPU not found — PhysX falls back to CPU / simulation hangs

**Symptoms:**
- Isaac Sim GPU table shows `Driver Version: 0`
- `[Error] [gpu.foundation.plugin] No device could be created`
- `[Error] [omni.kit.renderer.plugin] GPU Foundation is not initialized!`
- `[Warning] [omni.physx.plugin] No suitable CUDA GPU was found!`
- `[Warning] [omni.physx.plugin] PhysX GPU solver pipeline failed, switching to software`
- Simulation appears to hang indefinitely after `Starting the simulation...`

**Root cause — two separate issues, both required:**

**Issue 1: `NVIDIA_DRIVER_CAPABILITIES` (primary)**

The `nvidia/cuda:*` base image sets `ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility` in its
Dockerfile. This environment variable tells the nvidia-container-toolkit which driver
libraries to bind-mount into the container at start time. With only `compute,utility`, the
toolkit mounts CUDA libraries but NOT the OpenGL/Vulkan libraries:

- `libGLX_nvidia.so.0` (Vulkan ICD for GLX) — NOT mounted
- `libEGL_nvidia.so.0` (Vulkan ICD for EGL) — NOT mounted

Isaac Sim needs Vulkan even in headless mode (`app.vulkan = true` in its `.kit` files).
Without `libGLX_nvidia.so.0` in the container, the Vulkan loader has no ICD and cannot
enumerate any GPU device, causing `Driver Version: 0` and `No device could be created`.

**Critical:** `NVIDIA_DRIVER_CAPABILITIES` must be passed as a Docker environment variable
(`-e NVIDIA_DRIVER_CAPABILITIES=all`) so the toolkit reads it during the container runtime
hook, before the container starts. Setting it via `export` inside `extra_docker_shell_script`
is too late.

**Issue 2: `--gpus=all` (prerequisite)**

Without `--gpus=all` the toolkit doesn't expose any GPU device files (`/dev/nvidia*`) and
doesn't mount any driver libraries at all, regardless of `NVIDIA_DRIVER_CAPABILITIES`.

**Fix:** Both flags are required in `extra_docker_arguments`:
```
["--ipc=host", "--gpus=all", "-e", "NVIDIA_DRIVER_CAPABILITIES=all"]
```

The NVIDIA Vulkan ICD JSON (`/usr/share/vulkan/icd.d/nvidia_icd.json`) is also written by
`extra_docker_shell_script` to tell `libvulkan1` where to find the ICD library. This is
needed because the `nvidia/cuda:*` image does not include this manifest file.

**Verification:** The diagnostic lines in `extra_docker_shell_script` will print:
- `nvidia-smi` output — confirms the GPU is visible and `--gpus=all` is working
- `NVIDIA_DRIVER_CAPABILITIES=all` — confirms the env var took effect
- `libGLX_nvidia` found in `/usr/lib/x86_64-linux-gnu/` — confirms Vulkan ICD is mounted

After the fix, the GPU table in the task log should show the actual driver version (e.g.,
`Driver Version: 570`) and neither the Vulkan error nor the PhysX software-fallback warning
should appear.

### Wrong Python version (3.12 instead of 3.11)

If `python_binary: "/usr/bin/python3.11"` doesn't take effect, verify the setting
appears in the agent's config dump as `agent.python_binary = /usr/bin/python3.11`
(not empty). Same root cause as above — `python_binary` must be directly under `agent`,
not under `agent.package_manager`. Note: uv downloads its own Python 3.11 managed
binary regardless, so the actual training venv will use 3.11 either way.
