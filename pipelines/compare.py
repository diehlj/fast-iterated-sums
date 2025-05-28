import pandas as pd

from shared.defintions import BASE_METRICS, ModelMode, ResnetMode
from shared.helpers import get_logger, read_data


def comparison_pipeline(data_name: str) -> None:
    """
    Compare the classification results from FIS with that of
    baseline (https://github.com/chenyaofo/pytorch-cifar-models).
    """
    logger = get_logger(__name__)
    logger.info("Comparison pipeline has started.")

    index = pd.MultiIndex.from_product(
        iterables=[[rm for rm in ResnetMode], [mm for mm in ModelMode]],
        # names=["ResNet", "Model"],
    )
    table = pd.DataFrame(
        columns=[
            "accuracy@1 (%)",
            "accuracy@5 (%)",
            "total_params (M)",
            "total_mult_adds (M)",
        ],
        index=index,
    )

    for rm in ResnetMode:
        print(f"Working on {rm}...")

        for mm in ModelMode:
            if mm == ModelMode.BASE:
                metrics = BASE_METRICS[data_name][rm]

            else:
                metrics = read_data(
                    fname=f"data_name={data_name}_resnet_mode={rm}_model_mode={mm}_history.pkl",
                    path="./history",
                )
                metrics["accuracy@1"] *= 100.0
                metrics["accuracy@5"] *= 100.0

            table.loc[(rm, mm), table.columns] = [
                metrics[col.split(" ")[0]] for col in table.columns
            ]

    print("Comparison table:\n")
    print(table)

    logger.info("Comparison pipeline finished successfully.")

    return None
