# Isaac Imitation Learning

Imitation learning with Isaac Lab/Sim and Robomimic (ACT, Diffusion Policy, BC).

This Isaac Lab extension trains and evaluates imitation learning policies in GPU-accelerated simulation. It augments Isaac Lab task environments with robomimic policy entry points and integrates ClearML for experiment tracking.

## Installation

**Prerequisites:** Ubuntu 24.04 or 22.04. Python 3.11 (required by isaaclab 2.3 and isaacsim 5.1). NVIDIA GPU with CUDA support.

1. Clone this repository:

    ```bash
    git clone https://github.com/chen-tianjian/isaac_imitation_learning.git
    cd isaac_imitation_learning
    ```

2. Create a Python 3.11 virtual environment and install (select one way below):

    **virtualenv / venv**
    ```bash
    python3.11 -m venv .venv
    source .venv/bin/activate
    pip install -e . --extra-index-url https://pypi.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu128
    ```

    **uv**
    ```bash
    uv venv --python 3.11 --seed
    source .venv/bin/activate
    uv pip install -e .
    ```

    **conda**
    ```bash
    conda create -n isaac python=3.11 -y
    conda activate isaac
    pip install -e . --extra-index-url https://pypi.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu128
    ```

3. (Optional) Set up ClearML for experiment tracking:

    ```bash
    clearml-init
    ```

    ClearML is installed automatically but only activates when configured. Without it, all training logs are saved locally.

## Usage

### Training

```bash
python scripts/robomimic/train.py --task=Isaac-Stack-Cube-Franka-IK-Rel-v0 --algo=act --dataset=/path/to/data.hdf5
python scripts/robomimic/train.py --task=Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0 --algo=diffusion_policy --dataset=/path/to/data.hdf5
```

Datasets can also be pulled from ClearML by `clearml://` URIs:

```bash
python scripts/robomimic/train.py --task=Isaac-Stack-Cube-Franka-IK-Rel-v0 --algo=act --dataset=clearml://<DATASET_ID>/data.hdf5
```

Key flags:
- `--dataset PATH` -- Local path or `clearml://<DATASET_ID>[/<filename>]`
- `--epochs N` -- Override epoch count
- `--normalize_training_actions` -- Normalize actions to [-1, 1]
- `--clearml_project NAME` -- ClearML project name
- `--remote` -- Execute on ClearML agent queue
- `--no_clearml` -- Disable ClearML even if configured

### Evaluation

```bash
python scripts/robomimic/play.py --task=Isaac-Stack-Cube-Franka-IK-Rel-v0 --checkpoint=/path/to/model.pth
python scripts/robomimic/robust_eval.py --task=Isaac-Stack-Cube-Franka-IK-Rel-v0 --input_dir=/path/to/checkpoints/
```

Both scripts support `clearml://` URIs to resolve checkpoints from ClearML:

```bash
# Auto-select the latest checkpoint by epoch number
python scripts/robomimic/play.py --task=Isaac-Stack-Cube-Franka-IK-Rel-v0 --checkpoint=clearml://<TASK_ID>

# Or specify a specific checkpoint artifact
python scripts/robomimic/play.py --task=Isaac-Stack-Cube-Franka-IK-Rel-v0 --checkpoint=clearml://<TASK_ID>/model_epoch_100.pth
```

### Dummy Agents

Useful for verifying environment configuration:

```bash
python scripts/zero_agent.py --task=Isaac-Stack-Cube-Franka-IK-Rel-v0
python scripts/random_agent.py --task=Isaac-Stack-Cube-Franka-IK-Rel-v0
```

### List Environments

```bash
python scripts/list_envs.py
python scripts/list_envs.py --keyword "Franka"
```

## Project Structure

```
scripts/
    robomimic/train.py          # Training with ACT, Diffusion Policy, BC
    robomimic/play.py           # Single-checkpoint evaluation
    robomimic/robust_eval.py    # Batch evaluation across checkpoints
    random_agent.py             # Random action agent
    zero_agent.py               # Zero action agent
    list_envs.py                # List registered environments
source/isaac_imitation_learning/
    isaac_imitation_learning/
        tasks/                  # Environment registration and configs
        utils/
            clearml_utils.py    # ClearML integration (conditional activation)
            policy_utils.py     # Action chunking helpers
tests/                          # Unit tests (run without Isaac Sim)
```

## Testing

```bash
python -m pytest tests/ -v
```

Tests run without Isaac Sim or ClearML credentials.

## License

BSD-3-Clause
