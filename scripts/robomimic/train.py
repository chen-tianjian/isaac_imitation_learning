# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# MIT License
#
# Copyright (c) 2021 Stanford Vision and Learning Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
The main entry point for training policies from pre-collected data.

This script loads dataset(s), creates a model based on the algorithm specified,
and trains the model. It supports training on various environments with multiple
algorithms from robomimic.

Args:
    algo: Name of the algorithm to run.
    task: Name of the environment.
    name: If provided, override the experiment name defined in the config.
    dataset: If provided, override the dataset path defined in the config.
    log_dir: Directory to save logs.
    normalize_training_actions: Whether to normalize actions in the training data.

This file has been modified from the original robomimic version to integrate with IsaacLab.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaac_imitation_learning.utils.clearml_utils import (
    add_clearml_args,
    close_clearml_task,
    connect_configuration_file,
    init_clearml_task,
    maybe_execute_remotely,
    resolve_dataset,
    upload_checkpoint,
    upload_videos_from_dir,
)

from isaaclab.app import AppLauncher

# --- Argparse (BEFORE AppLauncher for ClearML remote execution support) ---
parser = argparse.ArgumentParser()

# Experiment Name (for tensorboard, saving models, etc.)
parser.add_argument(
    "--name",
    type=str,
    default=None,
    help="(optional) if provided, override the experiment name defined in the config",
)

# Dataset path, to override the one in the config
parser.add_argument(
    "--dataset",
    type=str,
    default=None,
    help="(optional) if provided, override the dataset path defined in the config",
)

parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--algo", type=str, default=None, help="Name of the algorithm.")
parser.add_argument("--log_dir", type=str, default="robomimic", help="Path to log directory")
parser.add_argument("--normalize_training_actions", action="store_true", default=False, help="Normalize actions")
parser.add_argument(
    "--epochs",
    type=int,
    default=None,
    help=(
        "Optional: Number of training epochs. If specified, overrides the number of epochs from the JSON training"
        " config."
    ),
)

add_clearml_args(parser)
args_cli = parser.parse_args()

# --- ClearML init (before AppLauncher) ---
clearml_task = init_clearml_task(
    args_cli, task_type="training", default_task_name=f"train_{args_cli.algo}_{args_cli.task}"
)
maybe_execute_remotely(clearml_task, args_cli)

# --- Resolve ClearML dataset URI (before AppLauncher, no sim needed) ---
if args_cli.dataset is not None:
    args_cli.dataset = resolve_dataset(args_cli.dataset)

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import datetime
import importlib
import json
import os
import shutil
import sys
import time
import traceback
from collections import OrderedDict

import gymnasium as gym
import h5py
import numpy as np
import psutil
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.train_utils as TrainUtils
import torch
from robomimic.algo import algo_factory
from robomimic.config import Config, config_factory
from robomimic.utils.log_utils import DataLogger, PrintLogger
from torch.utils.data import DataLoader

import isaac_imitation_learning.tasks  # noqa: F401 - triggers env spec augmentation


def get_env_metadata_from_dataset_isaaclab(dataset_path: str, set_env_specific_obs_processors: bool = True) -> dict:
    """Get environment metadata from dataset, handling Isaac Lab format.

    Isaac Lab datasets may use 'sim_args' instead of 'env_kwargs'. This function
    handles both formats for compatibility with robomimic v0.5.0.

    Args:
        dataset_path: Path to the HDF5 dataset.
        set_env_specific_obs_processors: Whether to set environment-specific observation processors.

    Returns:
        Environment metadata dictionary with 'env_kwargs' key (even if empty).
    """
    dataset_path = os.path.expanduser(dataset_path)
    with h5py.File(dataset_path, "r") as f:
        env_args_raw = f["data"].attrs["env_args"]
        # Handle bytes encoding for robustness
        if isinstance(env_args_raw, bytes):
            env_args = env_args_raw.decode("utf-8")
        else:
            env_args = str(env_args_raw)
        env_meta = json.loads(env_args)

    # Ensure env_kwargs exists for robomimic v0.5.0 compatibility
    env_kwargs = env_meta.setdefault("env_kwargs", {})

    # Remove env_lang if present (robomimic v0.5.0 does this)
    env_kwargs.pop("env_lang", None)

    if set_env_specific_obs_processors:
        EnvUtils.set_env_specific_obs_processing(env_meta=env_meta)

    return env_meta


def ensure_dataset_cfg_list(dataset_cfg):
    """Return dataset config as a validated single-item list.

    Handles multiple input formats (string, dict, list) for flexibility and provides
    clear error messages for invalid configurations.

    Args:
        dataset_cfg: Dataset configuration in various formats.

    Returns:
        List of dataset config dicts with validated "path" keys.

    Raises:
        TypeError: If dataset_cfg is not a supported type.
        ValueError: If dataset_cfg list is empty.
        NotImplementedError: If multiple datasets are provided.
        KeyError: If dataset config is missing "path" key.
    """
    if isinstance(dataset_cfg, str):
        dataset_cfg_list = [{"path": dataset_cfg}]
    elif isinstance(dataset_cfg, dict):
        dataset_cfg_list = [dataset_cfg]
    elif isinstance(dataset_cfg, list):
        dataset_cfg_list = dataset_cfg
    else:
        raise TypeError(f"Unsupported config.train.data type: {type(dataset_cfg)}")

    if len(dataset_cfg_list) == 0:
        raise ValueError("config.train.data list is empty.")
    if len(dataset_cfg_list) > 1:
        raise NotImplementedError("Multiple datasets are not currently supported in this training script.")
    if "path" not in dataset_cfg_list[0]:
        raise KeyError("Dataset config is missing required 'path' key.")
    return dataset_cfg_list



def normalize_hdf5_actions(config: Config, log_dir: str) -> str:
    """Normalizes actions in hdf5 dataset to [-1, 1] range.

    Args:
        config: The configuration object containing dataset path.
        log_dir: Directory to save normalization parameters.

    Returns:
        Path to the normalized dataset.
    """
    base, ext = os.path.splitext(config.train.data)
    normalized_path = base + "_normalized" + ext

    # Copy the original dataset
    print(f"Creating normalized dataset at {normalized_path}")
    shutil.copyfile(config.train.data, normalized_path)

    # Open the new dataset and normalize the actions
    with h5py.File(normalized_path, "r+") as f:
        dataset_paths = [f"/data/demo_{str(i)}/actions" for i in range(len(f["data"].keys()))]

        # Compute the min and max of the dataset
        dataset = np.array(f[dataset_paths[0]]).flatten()
        for i, path in enumerate(dataset_paths):
            if i != 0:
                data = np.array(f[path]).flatten()
                dataset = np.append(dataset, data)

        max = np.max(dataset)
        min = np.min(dataset)

        # Normalize the actions
        for i, path in enumerate(dataset_paths):
            data = np.array(f[path])
            normalized_data = 2 * ((data - min) / (max - min)) - 1  # Scale to [-1, 1] range
            del f[path]
            f[path] = normalized_data

        # Save the min and max values to log directory
        with open(os.path.join(log_dir, "normalization_params.txt"), "w") as f:
            f.write(f"min: {min}\n")
            f.write(f"max: {max}\n")

    return normalized_path


def train(config: Config, device: str, log_dir: str, ckpt_dir: str, video_dir: str, clearml_task=None):
    """Train a model using the algorithm specified in config.

    Args:
        config: Configuration object.
        device: PyTorch device to use for training.
        log_dir: Directory to save logs.
        ckpt_dir: Directory to save checkpoints.
        video_dir: Directory to save videos.
        clearml_task: ClearML Task instance (or None if ClearML is disabled/unavailable).
    """
    # first set seeds
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    print("\n============= New Training Run with Config =============")
    print(config)
    print("")

    print(f">>> Saving logs into directory: {log_dir}")
    print(f">>> Saving checkpoints into directory: {ckpt_dir}")
    print(f">>> Saving videos into directory: {video_dir}")

    if config.experiment.logging.terminal_output_to_txt:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, "log.txt"))
        sys.stdout = logger
        sys.stderr = logger

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # Validate and normalize dataset config format
    dataset_cfg_list = ensure_dataset_cfg_list(config.train.data)
    dataset_cfg = dataset_cfg_list[0]

    # make sure the dataset exists
    dataset_path = os.path.expanduser(str(dataset_cfg["path"]))
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset at provided path {dataset_path} not found!")

    # load basic metadata from training file
    print("\n============= Loaded Environment Metadata =============")
    env_meta = get_env_metadata_from_dataset_isaaclab(dataset_path=dataset_path)
    # robomimic v0.5.0 changed the function signature
    action_keys = getattr(config.train, "action_keys", ["actions"])
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_config=dataset_cfg, action_keys=action_keys, all_obs_keys=config.all_obs_keys, verbose=True
    )

    if config.experiment.env is not None:
        env_meta["env_name"] = config.experiment.env
        print("=" * 30 + "\n" + "Replacing Env to {}\n".format(env_meta["env_name"]) + "=" * 30)

    # create environment
    envs = OrderedDict()
    if config.experiment.rollout.enabled:
        # create environments for validation runs
        env_names = [env_meta["env_name"]]

        if config.experiment.additional_envs is not None:
            for name in config.experiment.additional_envs:
                env_names.append(name)

        for env_name in env_names:
            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta,
                env_name=env_name,
                render=False,
                render_offscreen=config.experiment.render_video,
                use_image_obs=shape_meta["use_images"],
            )
            envs[env.name] = env
            print(envs[env.name])

    print("")

    # robomimic v0.5.0 expects config.train.data to be a list of dicts with "path" keys
    # Convert string path to list format if needed
    with config.values_unlocked():
        if isinstance(config.train.data, str):
            config.train.data = [{"path": config.train.data}]

    # load training data (must be done before algo_factory in robomimic v0.5.0)
    trainset, validset = TrainUtils.load_data_for_training(config, obs_keys=shape_meta["all_obs_keys"])
    train_sampler = trainset.get_dataset_sampler()
    print("\n============= Training Dataset =============")
    print(trainset)
    print("")

    # robomimic v0.5.0 requires num_train_batches and num_epochs in optim_params for LR scheduler
    train_num_steps = config.experiment.epoch_every_n_steps
    with config.values_unlocked():
        if "optim_params" in config.algo:
            for k in config.algo.optim_params:
                config.algo.optim_params[k]["num_train_batches"] = (
                    len(trainset) if train_num_steps is None else train_num_steps
                )
                config.algo.optim_params[k]["num_epochs"] = config.train.num_epochs

    # setup for a new training run
    data_logger = DataLogger(log_dir, config=config, log_tb=config.experiment.logging.log_tb)
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )

    # save the config as a json file
    config_json_path = os.path.join(log_dir, "..", "config.json")
    with open(config_json_path, "w") as outfile:
        json.dump(config, outfile, indent=4)
    connect_configuration_file(clearml_task, config_json_path, "robomimic_config_file")

    print("\n============= Model Summary =============")
    print(model)  # print model summary
    print("")

    # maybe retrieve statistics for normalizing observations
    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        obs_normalization_stats = trainset.get_obs_normalization_stats()

    # initialize data loaders
    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.train.num_data_workers,
        drop_last=True,
    )

    if config.experiment.validate:
        # cap num workers for validation dataset at 1
        num_workers = min(config.train.num_data_workers, 1)
        valid_sampler = validset.get_dataset_sampler()
        valid_loader = DataLoader(
            dataset=validset,
            sampler=valid_sampler,
            batch_size=config.train.batch_size,
            shuffle=(valid_sampler is None),
            num_workers=num_workers,
            drop_last=True,
        )
    else:
        valid_loader = None

    # main training loop
    best_valid_loss = None
    last_ckpt_time = time.time()

    # number of learning steps per epoch (defaults to a full dataset pass)
    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps

    for epoch in range(1, config.train.num_epochs + 1):  # epoch numbers start at 1
        step_log = TrainUtils.run_epoch(model=model, data_loader=train_loader, epoch=epoch, num_steps=train_num_steps)
        model.on_epoch_end(epoch)

        # setup checkpoint path
        epoch_ckpt_name = f"model_epoch_{epoch}"

        # check for recurring checkpoint saving conditions
        should_save_ckpt = False
        if config.experiment.save.enabled:
            time_check = (config.experiment.save.every_n_seconds is not None) and (
                time.time() - last_ckpt_time > config.experiment.save.every_n_seconds
            )
            epoch_check = (
                (config.experiment.save.every_n_epochs is not None)
                and (epoch > 0)
                and (epoch % config.experiment.save.every_n_epochs == 0)
            )
            epoch_list_check = epoch in config.experiment.save.epochs
            last_epoch_check = epoch == config.train.num_epochs
            should_save_ckpt = time_check or epoch_check or epoch_list_check or last_epoch_check
        ckpt_reason = None
        if should_save_ckpt:
            last_ckpt_time = time.time()
            ckpt_reason = "time"

        print(f"Train Epoch {epoch}")
        print(json.dumps(step_log, sort_keys=True, indent=4))
        for k, v in step_log.items():
            if k.startswith("Time_"):
                data_logger.record(f"Timing_Stats/Train_{k[5:]}", v, epoch)
            else:
                data_logger.record(f"Train/{k}", v, epoch)

        # Evaluate the model on validation set
        if config.experiment.validate:
            with torch.no_grad():
                step_log = TrainUtils.run_epoch(
                    model=model, data_loader=valid_loader, epoch=epoch, validate=True, num_steps=valid_num_steps
                )
            for k, v in step_log.items():
                if k.startswith("Time_"):
                    data_logger.record(f"Timing_Stats/Valid_{k[5:]}", v, epoch)
                else:
                    data_logger.record(f"Valid/{k}", v, epoch)

            print(f"Validation Epoch {epoch}")
            print(json.dumps(step_log, sort_keys=True, indent=4))

            # save checkpoint if achieve new best validation loss
            valid_check = "Loss" in step_log
            if valid_check and (best_valid_loss is None or (step_log["Loss"] <= best_valid_loss)):
                best_valid_loss = step_log["Loss"]
                if config.experiment.save.enabled and config.experiment.save.on_best_validation:
                    epoch_ckpt_name += f"_best_validation_{best_valid_loss}"
                    should_save_ckpt = True
                    ckpt_reason = "valid" if ckpt_reason is None else ckpt_reason

        # Save model checkpoints based on conditions (success rate, validation loss, etc)
        if should_save_ckpt:
            ckpt_path = os.path.join(ckpt_dir, epoch_ckpt_name + ".pth")
            TrainUtils.save_model(
                model=model,
                config=config,
                env_meta=env_meta,
                shape_meta=shape_meta,
                ckpt_path=ckpt_path,
                obs_normalization_stats=obs_normalization_stats,
            )
            upload_checkpoint(clearml_task, ckpt_path)

        # Finally, log memory usage in MB
        process = psutil.Process(os.getpid())
        mem_usage = int(process.memory_info().rss / 1000000)
        data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
        print(f"\nEpoch {epoch} Memory Usage: {mem_usage} MB\n")

    # terminate logging
    data_logger.close()

    # Upload training videos to ClearML
    upload_videos_from_dir(clearml_task, video_dir, "training_rollouts")


def main(args: argparse.Namespace, clearml_task=None):
    """Train a model on a task using a specified algorithm.

    Args:
        args: Command line arguments.
        clearml_task: ClearML Task instance (or None if ClearML is disabled/unavailable).
    """
    # load config
    if args.task is not None:
        # obtain the configuration entry point
        cfg_entry_point_key = f"robomimic_{args.algo}_cfg_entry_point"
        task_name = args.task.split(":")[-1]

        print(f"Loading configuration for task: {task_name}")
        print(gym.envs.registry.keys())
        print(" ")
        cfg_entry_point_file = gym.spec(task_name).kwargs.pop(cfg_entry_point_key)
        # check if entry point exists
        if cfg_entry_point_file is None:
            raise ValueError(
                f"Could not find configuration for the environment: '{task_name}'."
                f" Please check that the gym registry has the entry point: '{cfg_entry_point_key}'."
            )

        # resolve module path if needed
        if ":" in cfg_entry_point_file:
            mod_name, file_name = cfg_entry_point_file.split(":")
            mod = importlib.import_module(mod_name)
            if mod.__file__ is None:
                raise ValueError(f"Could not find module file for: '{mod_name}'")
            mod_path = os.path.dirname(mod.__file__)
            config_file = os.path.join(mod_path, file_name)
        else:
            config_file = cfg_entry_point_file

        with open(config_file) as f:
            ext_cfg = json.load(f)
            config = config_factory(ext_cfg["algo_name"])
        # update config with external json - this will throw errors if
        # the external config has keys not present in the base algo config
        with config.values_unlocked():
            config.update(ext_cfg)
    else:
        raise ValueError("Please provide a task name through CLI arguments.")

    if args.dataset is not None:
        config.train.data = args.dataset

    if args.name is not None:
        config.experiment.name = args.name

    if args.epochs is not None:
        config.train.num_epochs = args.epochs

    # change location of experiment directory
    config.train.output_dir = os.path.abspath(os.path.join("./logs", args.log_dir, args.task))

    # Convert dataset config to list format expected by robomimic v0.5.0
    dataset_cfg_list = ensure_dataset_cfg_list(config.train.data)
    with config.values_unlocked():
        config.train.data = dataset_cfg_list

    # Create experiment directories (never overwrite — always create a new timestamped subdir)
    base_output_dir = os.path.join(
        os.path.abspath(config.train.output_dir),
        config.experiment.name,
        datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
    )
    log_dir = os.path.join(base_output_dir, "logs")
    ckpt_dir = os.path.join(base_output_dir, "models") if config.experiment.save.enabled else None
    video_dir = os.path.join(base_output_dir, "videos")
    os.makedirs(log_dir)
    if ckpt_dir:
        os.makedirs(ckpt_dir)
    os.makedirs(video_dir)

    if args.normalize_training_actions:
        config.train.data = normalize_hdf5_actions(config, log_dir)
        norm_params_path = os.path.join(log_dir, "normalization_params.txt")
        connect_configuration_file(clearml_task, norm_params_path, "normalization_params")

    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    config.lock()

    # catch error during training and print it
    res_str = "finished run successfully!"
    try:
        train(config, device, log_dir, ckpt_dir, video_dir, clearml_task=clearml_task)
    except Exception as e:
        res_str = f"run failed with error:\n{e}\n\n{traceback.format_exc()}"
    print(res_str)


if __name__ == "__main__":
    # run training
    main(args_cli, clearml_task=clearml_task)
    # close ClearML task before Isaac Sim shutdown to ensure proper finalization
    close_clearml_task(clearml_task)
    # close sim app
    simulation_app.close()
