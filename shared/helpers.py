import logging
import os
import pickle
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def get_rcparams():
    SMALL_SIZE = 9
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 11

    FONT = {"family": "serif", "serif": ["Palatino"], "size": SMALL_SIZE}

    rc_params = {
        "axes.titlesize": MEDIUM_SIZE,
        "axes.labelsize": SMALL_SIZE,
        "xtick.labelsize": SMALL_SIZE,
        "ytick.labelsize": SMALL_SIZE,
        "legend.fontsize": SMALL_SIZE,
        "figure.titlesize": BIGGER_SIZE,
        "font.family": FONT["family"],
        "font.serif": FONT["serif"],
        "font.size": FONT["size"],
        "text.usetex": True,
        "figure.constrained_layout.use": True,
        # "axes.spines.top": False,
        # "axes.spines.right": False,
        # "axes.spines.left": True,
        # "axes.spines.bottom": True,
        "xtick.bottom": True,
        "ytick.left": True,
        "axes.grid": False,
    }

    return rc_params


def read_data(fname: str, path: str) -> Any:
    """
    Function that reads .pkl file from a
    a given folder.

    Args:
    ----
        fname (str):  file name
        path (str): path to folder

    Returns:
    -------
            loaded file.
    """
    # Load pickle data
    with open(os.path.join(path, fname), "rb") as fp:
        loaded_file = pickle.load(fp)

    return loaded_file


def dump_data(data: Any, fname: str, path: str) -> None:
    """
    Function that dumps a pickled data into
    a specified path

    Args:
    ----
        data (Any): data to be pickled
        fname (str):  file name
        path (str): path to folder

    Returns:
    -------
            None
    """
    with open(os.path.join(path, fname), "wb") as fp:
        pickle.dump(data, fp)

    return None


class CustomFormatter(logging.Formatter):
    purple = "\x1b[1;35m"
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(levelname)s: [%(name)s] %(message)s"
    # "[%(asctime)s] (%(name)s) %(levelname)s: %(message)s"
    # "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: purple + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger(root_logger: str) -> logging.Logger:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CustomFormatter())

    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler],
    )

    return logging.getLogger(root_logger)


def create_classification_report(
    history: dict,
    n_params: int,
    y_test: list[int],
    y_pred: list[int],
    target_names: list[str],
) -> None:
    print("\n------------------ Model info -----------------\n")
    print(f"Number of model parameters: {n_params}")
    print(f"Wall clock time (WCT): {np.sum(history['wall_clock_time']):.2f} s")
    print(f"WCT/Epoch: {np.mean(history['wall_clock_time']):.2f} s")

    print("\n------------------ Confusion Matrix -----------------\n")
    print(confusion_matrix(y_test, y_pred))

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.2f}\n")

    print(f"Micro Precision: {precision_score(y_test, y_pred, average='micro'):.2f}")
    print(f"Micro Recall: {recall_score(y_test, y_pred, average='micro'):.2f}")
    print(f"Micro F1-score: {f1_score(y_test, y_pred, average='micro'):.2f}\n")

    print(f"Macro Precision: {precision_score(y_test, y_pred, average='macro'):.2f}")
    print(f"Macro Recall: {recall_score(y_test, y_pred, average='macro'):.2f}")
    print(f"Macro F1-score: {f1_score(y_test, y_pred, average='macro'):.2f}\n")

    print(
        f"Weighted Precision: {precision_score(y_test, y_pred, average='weighted'):.2f}"
    )
    print(f"Weighted Recall: {recall_score(y_test, y_pred, average='weighted'):.2f}")
    print(f"Weighted F1-score: {f1_score(y_test, y_pred, average='weighted'):.2f}")

    print("\n--------------- Classification Report ---------------\n")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("---------------------- FastIteratedSumsModel ----------------------")


@torch.no_grad()
def calculate_topk_accuracy(
    y_pred: torch.Tensor, y_true: torch.Tensor, topk: int = 1
) -> float:
    _, indices = y_pred.topk(k=topk, dim=-1, largest=True, sorted=True)
    y_pred = indices.t()  # (sz,topk) -> (topk, sz)

    # repeat same labels topk times to match result's shape
    y_true = y_true.view(1, -1)  # (sz) -> (1,sz)
    y_true = y_true.expand_as(y_pred)  # (1,sz) -> (topk,sz)

    correct = y_pred == y_true  # (topk,sz) of bool vals
    correct = correct.flatten()  # (topk*sz) of bool vals
    correct = correct.float()  # (topk*sz) of 1s or 0s
    correct = correct.sum()  # counts 1s (correct guesses)

    return correct.item()
