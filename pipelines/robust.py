import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

from shared.defintions import AEBackbone, MVTECCategory
from shared.helpers import (
    dump_data,
    get_logger,
)
from steps import (
    build_autoencoder,
    build_feature_extractor,
    build_model,
    build_optimizer,
    evaluate_anomaly_detector,
    load_data,
    load_mvtec,
    train_autoencoder,
    train_epoch,
)

load_dotenv()


def random_robustness_pipeline(
    data_name: str,
    resnet_mode: str,
    model_mode: str,
    semiring: str,
    hyperparams: dict,
    num_epochs: int,
) -> None:
    logger = get_logger(__name__)

    logger.info("Random robustness pipeline has started.")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    history = {"accuracy@1": [], "accuracy@5": []}

    ARTIFACT_TAG = (
        f"data_name={data_name}_resnet_mode={resnet_mode}_model_mode={model_mode}"
    )

    train_loader, validation_loader = load_data(
        data_name=data_name, batch_size=hyperparams["batch_size"]
    )

    for i in range(hyperparams["num_repeat"]):
        best_top1_val_acc = 0.0
        best_top5_val_acc = 0.0

        model = build_model(
            model_mode=model_mode,
            data_name=data_name,
            resnet_mode=resnet_mode,
            pool_mode=hyperparams["pool_mode"],
            num_nodes=hyperparams["num_nodes"],
            seed=None,  # setting seed to None to have a diferent tree architecture for each repetition
            semiring=semiring,
            tree_type="random",
        )
        model.to(device=DEVICE)
        # TODO: add functionalty for more than one GPU.

        optimizer, scheduler = build_optimizer(model=model, num_epochs=num_epochs)
        criterion = torch.nn.CrossEntropyLoss()

        for _ in range(num_epochs):
            metrics = train_epoch(
                train_loader=train_loader,
                validation_loader=validation_loader,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                device=DEVICE,
            )

            if metrics.top1_val_acc > best_top1_val_acc:
                best_top1_val_acc = metrics.top1_val_acc

            if metrics.top5_val_acc > best_top5_val_acc:
                best_top5_val_acc = metrics.top5_val_acc

        history["accuracy@1"].append(best_top1_val_acc)
        history["accuracy@5"].append(best_top5_val_acc)

        print(
            f"[Repeat {i + 1}/{hyperparams['num_repeat']}] accuracy@1={best_top1_val_acc}, accuracy@5={best_top5_val_acc}"
        )

    dump_data(
        data=history,
        fname=f"{ARTIFACT_TAG}_random_robustness_history.pkl",
        path="./history",
    )

    logger.info(
        "Random robustness pipeline finished successfully. See ./history dir to see the pickled pipeline history."
    )

    return None


def backbone_robustness_pipeline(
    semiring: str,
    tree_type: str,
    hyperparams: dict,
    num_epochs: int,
) -> None:
    logger = get_logger(__name__)
    logger.info("Anomaly detection robustness to resnet backbone pipeline has started.")

    results = {}
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for backbone in AEBackbone:
        auc_roc_scores = []

        for category in MVTECCategory:
            logger.info(f"Running with backbone={backbone}, category={category}...")

            loader = load_mvtec(
                batch_size=hyperparams["batch_size"],
                category=category,
                include_val=True,
            )

            fe = build_feature_extractor(backbone=backbone)
            fe.to(DEVICE)

            ae = build_autoencoder(
                in_channels=fe.OUT_CHANNELS,
                latent_dim=hyperparams["latent_dim"],
                num_nodes=hyperparams["num_nodes"],
                semiring=semiring,
                tree_type=tree_type,
                seed=hyperparams["seed"],
            )
            ae.to(DEVICE)

            _, _ = train_autoencoder(
                ae=ae,
                fe=fe,
                train_loader=loader.train_loader,
                validation_loader=loader.validation_loader,
                num_epochs=num_epochs,
                device=DEVICE,
            )

            evaluator = evaluate_anomaly_detector(
                category=category,
                transform=loader.transform,
                img_size=loader.img_size,
                fe=fe,
                ae=ae,
                device=DEVICE,
            )

            auc_roc_scores.append(evaluator.auc_roc_score * 100.0)

        mean_score = np.round(np.mean(auc_roc_scores), 1)
        std_score = np.round(np.std(auc_roc_scores), 1)

        results[backbone.value] = {
            "mean_auc_roc_score (%)": mean_score,
            "std_auc_roc_score (%)": std_score,
        }

        print(
            f"backbone: {backbone}, mean_auc_roc_score: {mean_score}%, std_auc_roc_score: {std_score}%"
        )

    results = pd.DataFrame(results).T
    results.index.name = "pretrained_model"
    print(results)

    dump_data(
        data=results,
        fname="ad_backbone_robustness_history.pkl",
        path="./history",
    )

    logger.info(
        "Pipeline finished successfully. See ./history dir to see the pickled pipeline history."
    )

    return None


def latent_dim_robustness_pipeline(
    semiring: str,
    tree_type: str,
    hyperparams: dict,
    num_epochs: int,
) -> None:
    logger = get_logger(__name__)
    logger.info("Anomaly detection robustness to latent dim pipeline has started.")

    results = {}
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    latent_dims = (4, 8, 16, 32)

    for dim in latent_dims:
        auc_roc_scores = []

        for category in MVTECCategory:
            logger.info(f"Running with latent_dim={dim}, category={category}...")

            loader = load_mvtec(
                batch_size=hyperparams["batch_size"],
                category=category,
                include_val=True,
            )

            fe = build_feature_extractor()
            fe.to(DEVICE)

            ae = build_autoencoder(
                in_channels=fe.OUT_CHANNELS,
                latent_dim=dim,
                num_nodes=hyperparams["num_nodes"],
                semiring=semiring,
                tree_type=tree_type,
                seed=hyperparams["seed"],
            )
            ae.to(DEVICE)

            _, _ = train_autoencoder(
                ae=ae,
                fe=fe,
                train_loader=loader.train_loader,
                validation_loader=loader.validation_loader,
                num_epochs=num_epochs,
                device=DEVICE,
            )

            evaluator = evaluate_anomaly_detector(
                category=category,
                transform=loader.transform,
                img_size=loader.img_size,
                fe=fe,
                ae=ae,
                device=DEVICE,
            )

            auc_roc_scores.append(evaluator.auc_roc_score * 100.0)

        mean_score = np.round(np.mean(auc_roc_scores), 1)
        std_score = np.round(np.std(auc_roc_scores), 1)

        results[dim] = {
            "mean_auc_roc_score (%)": mean_score,
            "std_auc_roc_score (%)": std_score,
        }

        print(
            f"latent_dim: {dim}, mean_auc_roc_score: {mean_score}%, std_auc_roc_score: {std_score}%"
        )

    results = pd.DataFrame(results).T
    results.index.name = "latent_dim"
    print(results)

    dump_data(
        data=results,
        fname="ad_latent_dim_robustness_history.pkl",
        path="./history",
    )

    logger.info(
        "Pipeline finished successfully. See ./history dir to see the pickled pipeline history."
    )

    return None


def topn_mean_robustness_pipeline(
    semiring: str,
    tree_type: str,
    hyperparams: dict,
    num_epochs: int,
) -> None:
    logger = get_logger(__name__)
    logger.info(
        "Anomaly detection robustness to top n error mean pipeline has started."
    )

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    topn_itr = (5, 10, 15, 20, 25, 30, 28 * 28)
    results = pd.DataFrame(columns=topn_itr, index=list(MVTECCategory))

    for category in MVTECCategory:
        auc_roc_scores = []
        logger.info(f"category={category}...")

        loader = load_mvtec(
            batch_size=hyperparams["batch_size"],
            category=category,
            include_val=True,
        )

        fe = build_feature_extractor()
        fe.to(DEVICE)

        ae = build_autoencoder(
            in_channels=fe.OUT_CHANNELS,
            latent_dim=hyperparams["latent_dim"],
            num_nodes=hyperparams["num_nodes"],
            semiring=semiring,
            tree_type=tree_type,
            seed=hyperparams["seed"],
        )
        ae.to(DEVICE)

        _, _ = train_autoencoder(
            ae=ae,
            fe=fe,
            train_loader=loader.train_loader,
            validation_loader=loader.validation_loader,
            num_epochs=num_epochs,
            device=DEVICE,
        )

        for topn in topn_itr:
            evaluator = evaluate_anomaly_detector(
                category=category,
                transform=loader.transform,
                img_size=loader.img_size,
                fe=fe,
                ae=ae,
                device=DEVICE,
                topn=topn,
            )

            auc_roc_scores.append(evaluator.auc_roc_score * 100.0)
            print(
                f"topn: {topn}, auc_roc_score: {evaluator.auc_roc_score * 100.0:.1f}%"
            )

        results.loc[category, topn_itr] = auc_roc_scores

    results = {
        k: {
            "mean_auc_roc_score (%)": np.round(m, 1),
            "std_auc_roc_score (%)": np.round(s, 1),
        }
        for k, m, s in zip(topn_itr, results.mean().values, results.std().values)
    }

    results = pd.DataFrame(results).T
    results.index.name = "topn"
    print(results)

    dump_data(
        data=results,
        fname="ad_topn_robustness_history.pkl",
        path="./history",
    )

    logger.info(
        "Pipeline finished successfully. See ./history dir to see the pickled pipeline history."
    )

    return None
