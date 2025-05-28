from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn import metrics
from torch import nn
from torch.utils.data import DataLoader

from shared.helpers import calculate_topk_accuracy


@dataclass(frozen=True)
class Evaluator:
    top1_accuracy: float
    top5_accuracy: float
    true_labels: list[int]
    predicted_labels: list[int]


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: int,
) -> Evaluator:
    model = model.to(device=device)
    model.eval()

    top1_accuracy = 0.0
    top5_accuracy = 0.0

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            true_labels.extend(labels.cpu().numpy().tolist())
            predicted_labels.extend(outputs.argmax(1).cpu().numpy().tolist())

            top1_accuracy += calculate_topk_accuracy(
                y_pred=outputs, y_true=labels, topk=1
            )
            top5_accuracy += calculate_topk_accuracy(
                y_pred=outputs, y_true=labels, topk=5
            )

    top1_accuracy /= len(test_loader.dataset)
    top5_accuracy /= len(test_loader.dataset)

    return Evaluator(
        top1_accuracy=top1_accuracy,
        top5_accuracy=top5_accuracy,
        true_labels=true_labels,
        predicted_labels=predicted_labels,
    )


@dataclass(frozen=True)
class AnomalyEvaluator:
    y_true: np.ndarray
    y_score: np.ndarray
    auc_roc_score: float
    thresholds: np.ndarray
    tpr: np.ndarray
    fpr: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    average_precision: float
    seg_map: dict[str, dict]


def calculate_anomaly_score(recon_error: torch.Tensor, topn: int) -> torch.Tensor:
    topn_mean = []

    for e in recon_error:
        sorted_vals, _ = torch.sort(e.reshape(-1), descending=True)

        topn_mean.append(sorted_vals[:topn].mean())

    return torch.stack(topn_mean)


def evaluate_anomaly_detector(
    category: str,
    transform: transforms.Compose,
    img_size: tuple[int, int],
    fe: nn.Module,
    ae: nn.Module,
    device: str,
    topn: int = 10,
):
    y_true = []
    y_score = []
    seg_map = {}

    ae.eval()

    test_path = Path(f"./data/mvtec/{category}/test")

    for path in test_path.glob("*/*.png"):
        img = Image.open(path).convert("RGB")
        data = transform(img).to(device).unsqueeze(0)

        with torch.no_grad():
            features = fe(data)
            reconstructions = ae(features)

        recon_error = ((features - reconstructions) ** 2).mean(dim=1)
        y_score.append(calculate_anomaly_score(recon_error, topn).cpu().numpy())

        fault_type = path.parts[-2]
        fname = path.parts[-1]
        seg_map[f"{category}+{fault_type}+{fname.split('.')[0]}"] = {
            "img": img,
            "map": nn.functional.interpolate(
                recon_error.unsqueeze(0), size=img_size, mode="bilinear"
            )  # this is done to match the image size
            .squeeze()
            .cpu()
            .numpy(),
        }

        y_true.append(0 if fault_type == "good" else 1)

    y_true = np.array(y_true)
    y_score = np.array(y_score).flatten()

    auc_roc_score = metrics.roc_auc_score(y_true=y_true, y_score=y_score)
    fpr, tpr, thresholds = metrics.roc_curve(
        y_true=y_true,
        y_score=y_score,
    )
    precision, recall, _ = metrics.precision_recall_curve(
        y_true=y_true,
        y_score=y_score,
    )
    average_precision = metrics.average_precision_score(
        y_true=y_true,
        y_score=y_score,
    )

    return AnomalyEvaluator(
        y_true=y_true,
        y_score=y_score,
        auc_roc_score=auc_roc_score,
        average_precision=average_precision,
        thresholds=thresholds,
        fpr=fpr,
        tpr=tpr,
        precision=precision,
        recall=recall,
        seg_map=seg_map,
    )
