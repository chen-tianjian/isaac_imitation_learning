# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared ClearML utility module for robomimic imitation learning scripts.

Provides ClearML integration with graceful degradation when ClearML is not installed.
All public functions accept task=None and gracefully no-op.

Key features:
    - Unified ``clearml://`` URI scheme for datasets, checkpoints, and input directories
    - Auto-detect ClearML availability; ``--no_clearml`` to disable
    - ``connect_configuration`` for config files (editable in ClearML UI)
    - ``report_media`` for inline video playback in ClearML "Debug Samples" tab
    - Continuous checkpoint upload as artifacts (crash resilient)
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile

# Lazy-loaded ClearML classes (set by resolve/upload functions when needed)
Task = None
Dataset = None

# Cached result of ClearML availability check
_clearml_available: bool | None = None


def is_clearml_available() -> bool:
    """Check if the clearml package is importable. Caches the result.

    Returns:
        True if clearml is installed and importable, False otherwise.
    """
    global _clearml_available
    if _clearml_available is None:
        try:
            import clearml  # noqa: F401

            _clearml_available = True
        except ImportError:
            _clearml_available = False
    return _clearml_available


def _ensure_clearml_imports():
    """Lazily import ClearML Task and Dataset classes.

    Raises:
        ImportError: If clearml is not installed.
    """
    global Task, Dataset
    if Task is None:
        from clearml import Dataset as _Dataset
        from clearml import Task as _Task

        Task = _Task
        Dataset = _Dataset


# ---------------------------------------------------------------------------
# CLI argument helpers
# ---------------------------------------------------------------------------


def add_clearml_args(parser):
    """Add ClearML-specific CLI arguments to an argparse parser.

    Args:
        parser: An argparse.ArgumentParser instance.
    """
    group = parser.add_argument_group("ClearML", "ClearML experiment tracking arguments")
    group.add_argument(
        "--clearml_project",
        type=str,
        default="IsaacLab/Robomimic",
        help="ClearML project name.",
    )
    group.add_argument(
        "--clearml_task_name",
        type=str,
        default=None,
        help="ClearML task name. Auto-generated if not provided.",
    )
    group.add_argument(
        "--clearml_queue",
        type=str,
        default="default",
        help="Queue name for remote execution.",
    )
    group.add_argument(
        "--remote",
        action="store_true",
        default=False,
        help="Enable ClearML remote execution via execute_remotely().",
    )
    group.add_argument(
        "--no_clearml",
        action="store_true",
        default=False,
        help="Disable ClearML even if installed.",
    )


# ---------------------------------------------------------------------------
# Task lifecycle
# ---------------------------------------------------------------------------


def init_clearml_task(args, task_type: str, default_task_name: str):
    """Initialize a ClearML Task if available and not disabled.

    Args:
        args: Parsed argparse namespace. Must have ``no_clearml``, ``clearml_project``,
            and ``clearml_task_name`` attributes.
        task_type: ClearML task type string (e.g. "training", "testing").
        default_task_name: Fallback task name if ``args.clearml_task_name`` is None.

    Returns:
        A ClearML Task instance, or None if ClearML is unavailable or disabled.
    """
    if getattr(args, "no_clearml", False) or not is_clearml_available():
        return None

    try:
        _ensure_clearml_imports()
        task_name = getattr(args, "clearml_task_name", None) or default_task_name
        task = Task.init(
            project_name=getattr(args, "clearml_project", "IsaacLab/Robomimic"),
            task_name=task_name,
            task_type=task_type,
            auto_connect_arg_parser=True,
            auto_connect_frameworks={"tensorboard": True},
        )
        return task
    except Exception as e:
        print(f"[ClearML] Warning: Failed to initialize ClearML task: {e}")
        return None


def maybe_execute_remotely(task, args):
    """If --remote is set, send the task to a ClearML queue and exit.

    Must be called BEFORE AppLauncher to avoid launching Isaac Sim locally
    when submitting remotely.

    Args:
        task: ClearML Task instance (or None).
        args: Parsed argparse namespace with ``remote`` and ``clearml_queue`` attributes.
    """
    if task is None or not getattr(args, "remote", False):
        return
    queue_name = getattr(args, "clearml_queue", "default")
    print(f"[ClearML] Sending task to queue '{queue_name}' for remote execution...")
    task.execute_remotely(queue_name=queue_name, exit_process=True)


def close_clearml_task(task):
    """Explicitly close and finalize a ClearML Task.

    Must be called BEFORE simulation_app.close() because Isaac Sim's shutdown
    may leave background threads that prevent Python's atexit handlers from
    completing, which ClearML relies on to mark tasks as "completed".

    Args:
        task: ClearML Task instance (or None for no-op).
    """
    if task is None:
        return
    task.close()


# ---------------------------------------------------------------------------
# URI helpers
# ---------------------------------------------------------------------------


def is_clearml_uri(path: str | None) -> bool:
    """Check if a path is a ``clearml://`` URI.

    Args:
        path: A file path or URI string, or None.

    Returns:
        True if path starts with ``clearml://``, False otherwise.
    """
    if path is None:
        return False
    return str(path).startswith("clearml://")


def parse_clearml_uri(uri: str) -> tuple[str, str | None]:
    """Parse a ``clearml://<id>[/<filename>]`` URI.

    Args:
        uri: A clearml:// URI string.

    Returns:
        Tuple of (resource_id, filename_or_None).

    Raises:
        ValueError: If the URI is malformed or empty.
    """
    if not uri or not uri.startswith("clearml://"):
        raise ValueError(f"Malformed ClearML URI: '{uri}'. Expected format: clearml://<id>[/<filename>]")

    remainder = uri[len("clearml://") :]
    if not remainder:
        raise ValueError(f"Malformed ClearML URI: '{uri}'. Resource ID is empty.")

    # Split on first slash to separate ID from optional filename
    parts = remainder.split("/", 1)
    resource_id = parts[0]
    filename = parts[1] if len(parts) > 1 else None

    if not resource_id:
        raise ValueError(f"Malformed ClearML URI: '{uri}'. Resource ID is empty.")

    return resource_id, filename


# ---------------------------------------------------------------------------
# Dataset / checkpoint resolution
# ---------------------------------------------------------------------------


def resolve_dataset(dataset_arg: str) -> str:
    """Resolve a --dataset argument to a local file path.

    Args:
        dataset_arg: Either a local path or a ``clearml://<dataset_id>[/<filename>]`` URI.

    Returns:
        Local path to the HDF5 dataset file.

    Raises:
        ValueError: If a ClearML dataset contains multiple HDF5 files and no filename is specified.
        FileNotFoundError: If the specified file is not found in the downloaded dataset.
    """
    if not is_clearml_uri(dataset_arg):
        return dataset_arg

    _ensure_clearml_imports()
    dataset_id, filename = parse_clearml_uri(dataset_arg)

    # Download dataset to local cache
    dataset = Dataset.get(dataset_id=dataset_id)
    local_dir = dataset.get_local_copy()

    if filename is not None:
        # Specific file requested
        target = os.path.join(local_dir, filename)
        if not os.path.exists(target):
            raise FileNotFoundError(
                f"File '{filename}' not found in ClearML dataset '{dataset_id}'. "
                f"Available files: {os.listdir(local_dir)}"
            )
        return target

    # No filename specified — find HDF5 files
    hdf5_files = []
    for root, _, files in os.walk(local_dir):
        for f in files:
            if f.endswith(".hdf5"):
                hdf5_files.append(os.path.join(root, f))

    if len(hdf5_files) == 1:
        return hdf5_files[0]
    elif len(hdf5_files) == 0:
        raise FileNotFoundError(f"No .hdf5 files found in ClearML dataset '{dataset_id}'.")
    else:
        filenames = [os.path.relpath(f, local_dir) for f in hdf5_files]
        raise ValueError(
            f"ClearML dataset '{dataset_id}' contains multiple .hdf5 files: {filenames}. "
            f"Please specify which file to use: clearml://{dataset_id}/<filename>"
        )


def resolve_checkpoint(checkpoint_arg: str) -> str:
    """Resolve a --checkpoint argument to a local file path.

    Args:
        checkpoint_arg: Either a local path or a ``clearml://<task_id>[/<artifact_name>]`` URI.

    Returns:
        Local path to the checkpoint file.

    Raises:
        FileNotFoundError: If no .pth artifacts are found or the specified artifact doesn't exist.
    """
    if not is_clearml_uri(checkpoint_arg):
        return checkpoint_arg

    _ensure_clearml_imports()
    task_id, artifact_name = parse_clearml_uri(checkpoint_arg)

    task = Task.get_task(task_id=task_id)

    if artifact_name is not None:
        # Download specific artifact
        if artifact_name not in task.artifacts:
            raise FileNotFoundError(
                f"Artifact '{artifact_name}' not found in ClearML task '{task_id}'. "
                f"Available artifacts: {list(task.artifacts.keys())}"
            )
        return task.artifacts[artifact_name].get_local_copy()

    # No specific artifact — find the latest .pth checkpoint by epoch number
    pth_artifacts = {k: v for k, v in task.artifacts.items() if k.endswith(".pth")}
    if not pth_artifacts:
        raise FileNotFoundError(f"No .pth artifacts found in ClearML task '{task_id}'.")

    # Sort by epoch number extracted from filename (e.g. model_epoch_150.pth -> 150)
    def _extract_epoch(name: str) -> int:
        match = re.search(r"epoch_(\d+)", name)
        return int(match.group(1)) if match else 0

    latest_name = max(pth_artifacts.keys(), key=_extract_epoch)
    return pth_artifacts[latest_name].get_local_copy()


def resolve_input_dir(input_dir_arg: str) -> str:
    """Resolve a --input_dir argument to a local directory path.

    Downloads all .pth artifacts from a ClearML task into a temporary directory.

    Args:
        input_dir_arg: Either a local path or a ``clearml://<task_id>`` URI.

    Returns:
        Local directory path containing the checkpoint files.

    Raises:
        FileNotFoundError: If no .pth artifacts are found.
    """
    if not is_clearml_uri(input_dir_arg):
        return input_dir_arg

    _ensure_clearml_imports()
    task_id, _ = parse_clearml_uri(input_dir_arg)

    task = Task.get_task(task_id=task_id)
    pth_artifacts = {k: v for k, v in task.artifacts.items() if k.endswith(".pth")}

    if not pth_artifacts:
        raise FileNotFoundError(f"No .pth artifacts found in ClearML task '{task_id}'.")

    # Download all checkpoints into a temp directory
    output_dir = tempfile.mkdtemp(prefix="clearml_checkpoints_")
    for name, artifact in pth_artifacts.items():
        local_path = artifact.get_local_copy()
        dest_path = os.path.join(output_dir, name)
        shutil.copy2(local_path, dest_path)

    return output_dir


# ---------------------------------------------------------------------------
# Upload / reporting
# ---------------------------------------------------------------------------


def upload_checkpoint(task, ckpt_path: str):
    """Upload a checkpoint file as a ClearML artifact.

    The artifact name is the filename (e.g. ``model_epoch_100.pth``).
    Called immediately after each save for crash resilience.

    Args:
        task: ClearML Task instance (or None for no-op).
        ckpt_path: Path to the checkpoint file.
    """
    if task is None:
        return
    name = os.path.basename(ckpt_path)
    task.upload_artifact(name=name, artifact_object=ckpt_path)


def connect_configuration_dict(task, config_dict, name: str):
    """Connect a dict configuration to a ClearML Task.

    Uses ``task.connect_configuration()`` so the config appears in the Configuration
    tab of the ClearML UI, is editable, and supports remote override.

    Args:
        task: ClearML Task instance (or None for no-op).
        config_dict: Configuration dictionary.
        name: Configuration section name in ClearML UI.
    """
    if task is None:
        return
    task.connect_configuration(config_dict, name=name)


def connect_configuration_file(task, file_path: str, name: str):
    """Connect a file-based configuration to a ClearML Task.

    Uses ``task.connect_configuration()`` with a file path so the file
    appears in the Configuration tab of the ClearML UI.

    Args:
        task: ClearML Task instance (or None for no-op).
        file_path: Path to the configuration file.
        name: Configuration section name in ClearML UI.
    """
    if task is None:
        return
    task.connect_configuration(file_path, name=name)


def report_scalar(task, title: str, series: str, value: float, iteration: int):
    """Report a scalar metric to ClearML.

    Args:
        task: ClearML Task instance (or None for no-op).
        title: Metric title (e.g. "evaluation").
        series: Metric series (e.g. "success_rate").
        value: Scalar value.
        iteration: Iteration/step number.
    """
    if task is None:
        return
    task.get_logger().report_scalar(title=title, series=series, value=value, iteration=iteration)


def report_evaluation_results(task, results_summary: dict, seed: int):
    """Report evaluation results (per-setting success rates) to ClearML.

    Reports each setting's best model success rate as a scalar, and uploads
    the full results summary as an artifact.

    Args:
        task: ClearML Task instance (or None for no-op).
        results_summary: Dict mapping setting names to dicts of model->success_rate.
        seed: The random seed used for this evaluation.
    """
    if task is None:
        return

    logger = task.get_logger()
    for setting, model_results in results_summary.items():
        if not model_results:
            continue
        best_model = max(model_results, key=model_results.get)
        best_rate = model_results[best_model]
        logger.report_scalar(title=f"eval_seed_{seed}", series=setting, value=best_rate, iteration=0)

    # Upload full results as artifact
    task.upload_artifact(name=f"eval_results_seed_{seed}", artifact_object=results_summary)


def report_video(task, video_path: str, title: str, series: str, iteration: int):
    """Upload a video to ClearML for inline playback in the Debug Samples tab.

    Args:
        task: ClearML Task instance (or None for no-op).
        video_path: Local path to the video file.
        title: Title for the media entry.
        series: Series name for the media entry.
        iteration: Iteration/step number.
    """
    if task is None:
        return
    task.get_logger().report_media(title=title, series=series, local_path=video_path, iteration=iteration)


def upload_videos_from_dir(task, video_dir: str, title: str):
    """Scan a directory for .mp4 files and report each via report_media().

    Args:
        task: ClearML Task instance (or None for no-op).
        video_dir: Directory to scan for .mp4 files.
        title: Title for the media entries in ClearML.
    """
    if task is None:
        return
    if not os.path.isdir(video_dir):
        return

    mp4_files = sorted(f for f in os.listdir(video_dir) if f.endswith(".mp4"))
    logger = task.get_logger()
    for i, filename in enumerate(mp4_files):
        video_path = os.path.join(video_dir, filename)
        logger.report_media(title=title, series=filename, local_path=video_path, iteration=i)
