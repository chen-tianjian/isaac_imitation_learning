# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Isaac Imitation Learning is an Isaac Lab extension for training and evaluating imitation learning policies (ACT, Diffusion Policy, BC) using robomimic. It extends Isaac Lab task environments with additional policy entry points and integrates ClearML for experiment tracking.

## Common Commands

### Installation
```bash
# With pip
pip install -e . --extra-index-url https://pypi.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu128

# With uv (index URLs are read from pyproject.toml)
uv pip install -e .
```

### Training
```bash
python scripts/robomimic/train.py --task=Isaac-Stack-Cube-Franka-IK-Rel-v0 --algo=act --dataset=/path/to/data.hdf5
python scripts/robomimic/train.py --task=Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0 --algo=diffusion_policy --dataset=/path/to/data.hdf5
```

Datasets can also be pulled from ClearML:
```bash
python scripts/robomimic/train.py --task=Isaac-Stack-Cube-Franka-IK-Rel-v0 --algo=act --dataset=clearml://<DATASET_ID>/data.hdf5
```

Key training flags:
- `--dataset PATH` - Local path or `clearml://<DATASET_ID>[/<filename>]`
- `--epochs N` - Override epoch count from config
- `--normalize_training_actions` - Normalize actions to [-1, 1]
- `--clearml_project NAME` - ClearML project for tracking
- `--remote` - Execute on ClearML queue
- `--no_clearml` - Disable ClearML

### Evaluation
```bash
python scripts/robomimic/play.py --task=Isaac-Stack-Cube-Franka-IK-Rel-v0 --checkpoint=/path/to/model.pth
python scripts/robomimic/robust_eval.py --task=Isaac-Stack-Cube-Franka-IK-Rel-v0 --input_dir=/path/to/checkpoints/
```

Both scripts support `clearml://` URIs:
```bash
# Auto-select the latest checkpoint by epoch number
python scripts/robomimic/play.py --task=Isaac-Stack-Cube-Franka-IK-Rel-v0 --checkpoint=clearml://<TASK_ID>

# Or specify a specific checkpoint artifact
python scripts/robomimic/play.py --task=Isaac-Stack-Cube-Franka-IK-Rel-v0 --checkpoint=clearml://<TASK_ID>/model_epoch_100.pth
```

### Dummy Agents
```bash
python scripts/zero_agent.py --task=Isaac-Stack-Cube-Franka-IK-Rel-v0
python scripts/random_agent.py --task=Isaac-Stack-Cube-Franka-IK-Rel-v0
```

### List Available Environments
```bash
python scripts/list_envs.py  # Lists Stack-Cube environments by default
python scripts/list_envs.py --keyword "Franka"
```

### Testing
```bash
python -m pytest tests/                        # Run all tests
python -m pytest tests/test_clearml_utils.py -v  # Single test file
```

### Code Formatting
```bash
pip install pre-commit
pre-commit run --all-files
```

## Architecture

### Environment Registration Pattern

The package uses lazy registration of gym environments:

1. `isaac_imitation_learning/__init__.py` - Does NOT trigger registration on import
2. `import isaac_imitation_learning.tasks` - Triggers registration (must be called AFTER AppLauncher)
3. `tasks/manipulation/stack/__init__.py` - Augments existing isaaclab_tasks specs with ACT/Diffusion Policy entry points

Entry points follow the pattern: `robomimic_{algo}_cfg_entry_point` pointing to JSON configs in `tasks/manipulation/stack/agents/robomimic/`.

### ClearML Integration

`utils/clearml_utils.py` provides graceful ClearML integration:
- All functions accept `task=None` and no-op safely
- ClearML is required dependency but only active when configured (`clearml-init` or env vars)
- `clearml://` URI scheme for datasets and checkpoints
- Remote execution support - `maybe_execute_remotely()` must be called BEFORE AppLauncher

### Training Script Flow

`scripts/robomimic/train.py` follows this order:
1. Parse args and init ClearML (before any Isaac Sim imports)
2. Resolve `clearml://` URIs
3. Launch AppLauncher (starts Isaac Sim)
4. Import `isaac_imitation_learning.tasks` to register environments
5. Load config, create environment, train model

### Policy Utilities

`utils/policy_utils.py` handles action-chunking policies (ACT, Diffusion Policy):
- `is_action_chunking_policy()` - Detects if policy needs frame stacking
- `get_observation_horizon()` - Gets required frame stack size
- `stack_observations()` - Prepares observations for inference

## Code Style

- Line length: 120
- Python: 3.11
- Docstrings: Google style
- Import order: standard-library → third-party → omniverse → isaaclab modules → first-party
- Linting: ruff with E, W, F, I, UP, C90, SIM, RET rules
- License: BSD-3-Clause

## Key Dependencies

- Isaac Sim (`isaacsim[all,extscache]==5.1.0`) — from NVIDIA PyPI
- PyTorch (`torch==2.7.0`, `torchvision==0.22.0`) — from PyTorch cu128 index
- Isaac Lab (`isaaclab==2.3.2`) — from NVIDIA PyPI
- robomimic fork: `git+https://github.com/chen-tianjian/robomimic.git@act`
- clearml (required, conditionally active)
- psutil
