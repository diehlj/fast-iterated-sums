import numpy as np
import pandas as pd

from shared.defintions import MVTECCategory, ResnetMode
from shared.helpers import get_logger, read_data
from shared.plotter import plot_ad_roc_curve, plot_seg_map


def ad_analysis_pipeline(ablation: bool):
    """
    Analyse the results from running the command:
    fis ad select [OPTIONS] [FLAG].
    """
    logger = get_logger(__name__)
    logger.info("Anomaly detection analysis pipeline has started.")

    auc_roc_scores = []
    for c in MVTECCategory:
        history = read_data(
            fname=f"ad_category={c}_ablation={ablation}_history.pkl",
            path="./history",
        )

        plot_seg_map(
            category=c,
            seg_map=history["seg_map"],
            y_score=history["y_score"],
            y_true=history["y_true"],
            thresholds=history["thresholds"],
            ablation=ablation,
        )

        auc_roc_scores.append(history["auc_roc_score"] * 100.0)

    auc_roc_scores.extend([np.mean(auc_roc_scores), np.std(auc_roc_scores)])

    results = pd.DataFrame(index=list(MVTECCategory) + ["mean", "std"])
    results["roc_auc_score (%)"] = auc_roc_scores
    results["roc_auc_score (%)"] = results["roc_auc_score (%)"].round(decimals=1)

    print(results)
    plot_ad_roc_curve(ablation=ablation)

    logger.info(
        "Anomaly detection analysis pipeline finished successfully. "
        "Please see the ./plots for the generated plots."
    )

    return None


def clf_analysis_pipeline(data_name: str, model_mode: str):
    """
    Analyse the results from the tree randomness robustness pipeline.
    """
    for rm in ResnetMode:
        artifact_tag = f"data_name={data_name}_resnet_mode={rm}_model_mode={model_mode}"

        history = read_data(
            fname=f"{artifact_tag}_random_robustness_history.pkl",
            path="./history",
        )

        print(
            f"{rm}: [acc@1: mean={np.mean(history['accuracy@1']) * 100.0:.2f}%, std={np.std(history['accuracy@1']) * 100.0:.2f}%] "
            f"[acc@5: mean={np.mean(history['accuracy@5']) * 100.0:.2f}%, std={np.std(history['accuracy@5']) * 100.0:.2f}%]"
        )

    return None
