import json
import math
import os
from typing import List

from transformers.trainer import TRAINER_STATE_NAME

from .logging import get_logger
from .packages import is_matplotlib_available


if is_matplotlib_available():
    import matplotlib.pyplot as plt


logger = get_logger(__name__)


def smooth(scalars: List[float]) -> List[float]:
    r"""
    EMA implementation according to TensorBoard.
    """
    last = scalars[0]
    smoothed = list()
    weight = 1.8 * (1 / (1 + math.exp(-0.05 * len(scalars))) - 0.5)  # a sigmoid function
    for next_val in scalars:
        smoothed_val = last * weight + (1 - weight) * next_val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_loss(save_dictionary: os.PathLike, keys: List[str] = ["loss"]) -> None:
    with open(os.path.join(save_dictionary, TRAINER_STATE_NAME), "r", encoding="utf-8") as f:
        data = json.load(f)

    for key in keys:
        steps, metrics = [], []
        for i in range(len(data["log_history"])):
            if key in data["log_history"][i]:
                steps.append(data["log_history"][i]["step"])
                metrics.append(data["log_history"][i][key])

        if len(metrics) == 0:
            logger.warning(f"No metric {key} to plot.")
            continue

        plt.figure()
        plt.plot(steps, metrics, color="#1f77b4", alpha=0.4, label="original")
        plt.plot(steps, smooth(metrics), color="#1f77b4", label="smoothed")
        plt.title("training {} of {}".format(key, save_dictionary))
        plt.xlabel("step")
        plt.ylabel(key)
        plt.legend()
        figure_path = os.path.join(save_dictionary, "training_{}.png".format(key.replace(os.path.sep, "_")))
        plt.savefig(figure_path, format="png", dpi=100)
        print("Figure saved at:", figure_path)


def save_loss_and_step(save_dictionary: os.PathLike) -> None:
    with open(os.path.join(save_dictionary, TRAINER_STATE_NAME), "r", encoding="utf-8") as f:
        data = json.load(f)

    eval_losses = []
    for log in data['log_history']:
        if 'eval_loss' in log:
            eval_losses.append((log['eval_loss'], log['step']))

    eval_losses.sort()

    with open(os.path.join(save_dictionary, 'sorted_eval_losses.txt'), 'w') as output_file:
        for loss, step in eval_losses:
            output_file.write(f"Step: {step}, Eval Loss: {loss}\n")

    print("Results have been written to sorted_eval_losses.txt")
