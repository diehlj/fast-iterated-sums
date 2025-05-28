from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

from shared.defintions import (
    COLOUR_MAP,
    LINE_MAP,
    DocType,
    ModelMode,
    MVTECCategory,
    ResnetMode,
)
from shared.helpers import get_rcparams, read_data

plt.rcParams.update(get_rcparams())


def set_size(
    width: float | str,
    fraction: float = 1.0,
    subplots: tuple = (1, 1),
    adjust_height: float | None = None,
) -> tuple:
    if width == DocType.THESIS:
        width_pt = 426.79135
    elif width == DocType.BEAMER:
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    if adjust_height is not None:
        golden_ratio += adjust_height

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def plot_training_pipeline_history(history: dict, artifact_tag: str) -> None:
    fig = plt.figure(figsize=set_size(subplots=(1, 2), width=DocType.THESIS))
    fig_labels = ("a", "b")
    # marker_style = dict(marker="o", ms=3, mfc="white")

    for i, m in enumerate(["loss", "accuracy"]):
        ax = fig.add_subplot(1, 2, i + 1)
        ax.text(
            x=-0.1,
            y=1.2,
            s=r"\bf {}".format(fig_labels[i]),
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
        )

        for d, c, style in zip(
            ["train", "validation"], ["darkcyan", "crimson"], ["-", "--"]
        ):
            ax.plot(
                history["epochs"],
                np.array(history[f"{d}_{m}"]) * 100.0
                if m == "accuracy"
                else history[f"{d}_{m}"],
                # **marker_style,
                color=c,
                label=f"{d}".capitalize(),
                linestyle=style,
            )
        ax.set_xlabel("Epochs")
        ax.set_ylabel(
            r"{} (\%)".format(m.capitalize()) if m == "accuracy" else m.capitalize()
        )
        ax.xaxis.set_major_locator(
            tck.MaxNLocator(integer=True, nbins=10, steps=[1, 5])
        )

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.2),
        frameon=False,
    )

    plt.savefig(
        fname=f"./plots/{artifact_tag}.pdf",
        bbox_inches="tight",
    )

    return None


def plot_confusion_matrix(
    true_labels: np.ndarray | list[int],
    predicted_labels: np.ndarray | list[int],
    classes: list[str],
    title: str,
    artifact_tag: str,
) -> None:
    cm = confusion_matrix(true_labels, predicted_labels)
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_percent = cm_percent * 100.0

    _, ax = plt.subplots(figsize=set_size(width=DocType.THESIS))

    ax = sns.heatmap(
        cm_percent,
        vmin=0.0,
        vmax=100.0,
        cmap=plt.cm.Reds,
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
        cbar=False,
        # cbar_kws={"label": r"Confusion bar (\%)"},
        annot=True,
        fmt=".2f",
    )

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(title)

    plt.yticks(rotation=0)
    plt.xticks(rotation=90)

    plt.savefig(f"./plots/{artifact_tag}.pdf", bbox_inches="tight")

    return None


def plot_metric_vs_epochs(metric: str) -> None:
    fig = plt.figure(figsize=set_size(subplots=(2, 2), width=DocType.THESIS))
    alphabet_tags = ("a", "b", "c", "d")

    IS_ACCURACY = "accuracy" in metric
    ylabel = metric.replace("_", " ").capitalize()

    if IS_ACCURACY:
        ylabel = r"{} (\%)".format(ylabel)

    for i, rm in enumerate(ResnetMode):
        ax = fig.add_subplot(2, 2, i + 1)
        ax.text(
            x=-0.1,
            y=1.25,
            s=r"\bf {}".format(alphabet_tags[i]),
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
        )

        for mm in ModelMode:
            history = read_data(
                fname=f"model_mode={mm}_resnet_mode={rm}_history.pkl", path="./history"
            )
            ax.plot(
                history["epochs"],
                np.array(history[metric]) * 100.0 if IS_ACCURACY else history[metric],
                label=mm.capitalize(),
                color=COLOUR_MAP[mm],
                linestyle=LINE_MAP[mm],
            )
        ax.set_xlabel("Epochs")

        if IS_ACCURACY:
            ax.set_ylim([None, 100.0])

        if i % 2 == 0:
            ax.set_ylabel(ylabel)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.15),
        frameon=False,
    )

    plt.savefig(
        fname=f"./plots/{metric}_vs_epochs_comparison.pdf",
        bbox_inches="tight",
    )

    return None


def plot_selected_metric_vs_epochs(metric: str) -> None:
    fig = plt.figure(
        figsize=set_size(subplots=(1, 3), width=DocType.THESIS, adjust_height=0.2)
    )
    alphabet_tags = ("a", "b", "c")
    ylabel = metric.replace("_", " ").capitalize()

    IS_ACCURACY = "accuracy" in metric

    if IS_ACCURACY:
        ylabel = r"{} (\%)".format(ylabel)

    for i, (d, b) in enumerate(
        zip(
            (ResnetMode.RESNET20, ResnetMode.RESNET32, ResnetMode.RESNET44),
            (ResnetMode.RESNET32, ResnetMode.RESNET44, ResnetMode.RESNET56),
        )
    ):
        ax = fig.add_subplot(1, 3, i + 1)
        ax.text(
            x=-0.1,
            y=1.25,
            s=r"\bf {}".format(alphabet_tags[i]),
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
        )

        downsample = read_data(
            fname=f"model_mode={ModelMode.DOWNSAMPLE}_resnet_mode={d}_history.pkl",
            path="./history",
        )
        base = read_data(
            fname=f"model_mode={ModelMode.BASE}_resnet_mode={b}_history.pkl",
            path="./history",
        )

        if IS_ACCURACY:
            downsample[metric] = np.array(downsample[metric]) * 100.0
            base[metric] = np.array(base[metric]) * 100.0
            ax.set_ylim([None, 100.0])

        ax.plot(
            downsample["epochs"],
            downsample[metric],
            label=f"{ModelMode.DOWNSAMPLE}-{d.split('t')[1]}".capitalize(),
            color=COLOUR_MAP[ModelMode.DOWNSAMPLE],
            linestyle=LINE_MAP[ModelMode.DOWNSAMPLE],
        )
        ax.plot(
            base["epochs"],
            base[metric],
            label=f"{ModelMode.BASE}-{b.split('t')[1]}".capitalize(),
            color=COLOUR_MAP[ModelMode.BASE],
            linestyle=LINE_MAP[ModelMode.BASE],
        )

        ax.legend(
            frameon=False,
        )
        ax.set_xlabel("Epochs")

        if i == 0:
            ax.set_ylabel(ylabel)

    plt.savefig(
        fname=f"./plots/selcted_{metric}_vs_epochs_comparison.pdf",
        bbox_inches="tight",
    )

    return None


def plot_image_samples(
    data_loader: DataLoader,
    tag: str,
    num_images: int = 8,
):
    images, _ = next(iter(data_loader))
    images = images[:num_images]
    images = images.cpu()

    fig = plt.figure(
        figsize=set_size(
            subplots=(2, num_images // 2),
            width=DocType.THESIS,
        )
    )

    for i in range(num_images):
        ax = fig.add_subplot(2, num_images // 2, i + 1)
        ax.imshow(images[i].permute(1, 2, 0))

        ax.axis("off")

    plt.savefig(f"./plots/{tag}.pdf")

    return None


def plot_anomaly_scores():
    num_category = len(MVTECCategory)
    fig = plt.figure(
        figsize=set_size(
            subplots=(num_category // 3, 3),
            width=DocType.THESIS,
        )
    )

    for i, category in enumerate(MVTECCategory):
        ax = fig.add_subplot(num_category // 3, 3, i + 1)

        history = read_data(
            fname=f"ad_category={category}_history.pkl", path="./history"
        )

        ax.hist(
            history["anomaly_scores"], bins=50, color="green", label="Anomaly scores"
        )
        ax.axvline(
            history["threshold"],
            color="red",
            linestyle="--",
            label="Anomaly threshold",
        )
        ax.set_title(label=category)

        if i % 3 == 0:
            ax.set_ylabel("Count")

        if i > 11:
            ax.set_xlabel("Anomaly scores")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.05),
        frameon=False,
    )

    plt.savefig(
        fname="./plots/ad_anomaly_scores.pdf",
        bbox_inches="tight",
    )

    return None


def plot_ad_roc_curve(ablation: bool) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=set_size(
            subplots=(1, 2),
            width=DocType.THESIS,
            adjust_height=0.3,
        ),
    )

    for category in MVTECCategory:
        history = read_data(
            fname=f"ad_category={category}_ablation={ablation}_history.pkl",
            path="./history",
        )

        axes[0].plot(
            history["fpr"],
            history["tpr"],
            label=r"{}: AUROC = {:.1f}\%, AP = {:.1f}\%".format(
                category.replace("_", " "),
                history["auc_roc_score"] * 100.0,
                history["average_precision"] * 100.0,
            ),
        )
        axes[0].set_ylabel("True positive rate")
        axes[0].set_xlabel("False positive rate")
        axes[0].set_title("ROC Curve")

        axes[1].plot(
            history["precision"],
            history["recall"],
        )
        axes[1].set_ylabel("Precision")
        axes[1].set_xlabel("Recall")
        axes[1].set_title("Precision-Recall Curve")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.25),
        frameon=False,
    )

    plt.savefig(
        fname=f"./plots/ad_roc_curve_ablation={ablation}.pdf",
        bbox_inches="tight",
    )
    return None


def plot_seg_map(
    category: str,
    seg_map: dict[str, Any],
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray,
    ablation: bool,
) -> None:
    test_path = Path(f"./data/mvtec/{category}/test")

    fault_types = []
    for path in test_path.iterdir():
        fault_types.append(path.parts[-1])

    fault_types = np.unique(fault_types)
    subplots = (3, len(fault_types))
    fig, axes = plt.subplots(
        subplots[0],
        subplots[1],
        figsize=set_size(subplots=subplots, width=DocType.THESIS, adjust_height=0.5),
    )

    f1_scores = [f1_score(y_true, y_score >= threshold) for threshold in thresholds]
    best_threshold = thresholds[np.argmax(f1_scores)]

    for i, fault in enumerate(fault_types):
        fault_history = {k: seg_map[k] for k in seg_map if f"{category}+{fault}" in k}
        random_key = np.random.choice(list(fault_history.keys()))

        axes[0, i].imshow(fault_history[random_key]["img"])
        axes[0, i].set_title(fault.replace("_", " "))
        axes[0, i].axis("off")

        axes[1, i].imshow(
            fault_history[random_key]["map"],
            cmap="jet",
            vmin=best_threshold,
            vmax=best_threshold * 2,
        )
        axes[1, i].axis("off")

        axes[2, i].imshow(
            (fault_history[random_key]["map"] >= best_threshold), cmap="gray"
        )
        axes[2, i].axis("off")

    fig.suptitle(category.title())
    plt.savefig(
        fname=f"./plots/{category}_seg_map_ablation={ablation}.pdf",
        bbox_inches="tight",
    )

    return None


# TODO: include confusion matrix: use f1 threshold
