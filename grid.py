from pipelines.train import ad_training_pipeline
from shared.params import AD_TRAIN_HYPERPARAMS


def main() -> None:
    for seed in range(1, 125):
        auc = ad_training_pipeline(
            category="grid",
            semiring="real",
            tree_type="parameter_sharing_v2",
            hyperparams=AD_TRAIN_HYPERPARAMS,
            seed=seed,
            num_epochs=200,
            ablation=False,
        )

        print(f"seed={seed}, auc={auc * 100:.1f}%")

    return None

if __name__ == "__main__":
    main()
