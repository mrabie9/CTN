import numpy as np
from typing import List, Dict, Any, Optional


def get_logger(keys: List[str], n_runs: int = 1, n_tasks: int = 1) -> List[Dict[str, Dict[str, np.ndarray]]]:
    """Create a nested logging structure.

    Parameters
    ----------
    keys : list of str
        Metrics to track during training/evaluation.
    n_runs : int, optional
        Number of experiment runs, by default 1.
    n_tasks : int, optional
        Total number of tasks, by default 1.

    Returns
    -------
    list
        A list where each element corresponds to a run and contains
        dictionaries for different modes (train/valid/test). Each mode
        stores an ``n_tasks x n_tasks`` matrix for every metric in
        ``keys``. The matrix is organised such that ``[task_t, task]``
        stores the metric for evaluating task ``task_t`` after learning
        task ``task``.
    """

    log: List[Dict[str, Dict[str, np.ndarray]]] = []
    modes = ["train", "valid", "test"]
    for _ in range(n_runs):
        run_dict: Dict[str, Dict[str, np.ndarray]] = {}
        for mode in modes:
            mode_dict: Dict[str, np.ndarray] = {}
            for k in keys:
                mode_dict[k] = np.zeros((n_tasks, n_tasks))
            run_dict[mode] = mode_dict
        log.append(run_dict)
    return log


def get_temp_logger(_wandb: Any, keys: List[str]) -> Dict[str, List[Any]]:
    """Return a simple logger used for accumulating batch statistics.

    Parameters
    ----------
    _wandb : Any
        Placeholder for compatibility with the expected API. It is not
        used but kept to mirror the original signature which accepts a
        wandb object.
    keys : list of str
        Metric names to track.

    Returns
    -------
    dict
        Dictionary mapping each metric name to an empty list where values
        can be appended.
    """

    return {k: [] for k in keys}


def logging_per_task(wandb: Any,
                     log: List[Dict[str, Dict[str, np.ndarray]]],
                     run: int,
                     mode: str,
                     metric: str,
                     task: int,
                     task_t: Optional[int] = None,
                     value: Optional[float] = None) -> None:
    """Store a metric in the logging structure and optionally report to wandb.

    Parameters
    ----------
    wandb : Any
        Weights & Biases run object. If ``None`` the function simply
        updates ``log``.
    log : list
        Logger returned by :func:`get_logger`.
    run : int
        Index of the current experiment run.
    mode : str
        One of ``'train'``, ``'valid'`` or ``'test'``.
    metric : str
        Metric name.
    task : int
        Index of the task after which the metric is measured.
    task_t : int, optional
        Task being evaluated. If ``None`` the metric is assumed to be a
        one-dimensional array indexed by ``task``.
    value : float, optional
        Metric value to record.
    """

    # Ensure mode exists
    if mode not in log[run]:
        log[run][mode] = {}

    mode_dict = log[run][mode]

    if task_t is not None:
        # Matrix-style metric
        if metric not in mode_dict:
            # infer n_tasks from existing metric if available
            example_key = next(iter(mode_dict)) if mode_dict else None
            if example_key is not None:
                n_tasks = mode_dict[example_key].shape[0]
            else:
                n_tasks = task + 1
            mode_dict[metric] = np.zeros((n_tasks, n_tasks))
        mode_dict[metric][task_t, task] = value if value is not None else 0.0
        if wandb is not None:
            wandb.log({f"{mode}/{metric}_t{task_t}_a{task}": value})
    else:
        # Vector-style metric
        if metric not in mode_dict:
            example_key = next(iter(mode_dict)) if mode_dict else None
            if example_key is not None:
                n_tasks = mode_dict[example_key].shape[0]
            else:
                n_tasks = task + 1
            mode_dict[metric] = np.zeros(n_tasks)
        mode_dict[metric][task] = value if value is not None else 0.0
        if wandb is not None:
            wandb.log({f"{mode}/{metric}_a{task}": value})
