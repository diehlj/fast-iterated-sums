import os
import time

import torch
from dotenv import load_dotenv
from torchinfo import summary

import wandb
from shared.helpers import (
    dump_data,
    get_logger,
)
from shared.plotter import plot_training_pipeline_history
from steps import (
    build_ablation_autoencoder,
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


def training_pipeline(
    data_name: str,
    model_mode: str,
    resnet_mode: str,
    semiring: str,
    tree_type: str,
    hyperparams: dict,
    num_epochs: int,
    include_wandb: bool,
) -> None:
    logger = get_logger(__name__)

    logger.info("Training pipeline has started.")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    history = {
        "epochs": [],
        "wall_clock_time": [],
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }
    best_top1_val_acc = 0.0
    best_top5_val_acc = 0.0
    best_epoch = 1

    ARTIFACT_TAG = (
        f"data_name={data_name}_resnet_mode={resnet_mode}_model_mode={model_mode}"
    )

    train_loader, validation_loader = load_data(
        data_name=data_name, batch_size=hyperparams["batch_size"]
    )

    model = build_model(
        model_mode=model_mode,
        data_name=data_name,
        resnet_mode=resnet_mode,
        pool_mode=hyperparams["pool_mode"],
        num_nodes=hyperparams["num_nodes"],
        seed=hyperparams["seed"],
        semiring=semiring,
        tree_type=tree_type,
    )
    model.to(device=DEVICE)
    # TODO: add functionalty for more than one GPU.

    logger.info("Getting model architecture...")
    _, C, H, W = next(iter(train_loader))[0].shape
    model_summary = summary(
        model,
        input_size=(1, C, H, W),
        verbose=0,
    )  # summary is calculated with batch_size = 1
    print(model_summary)
    history["total_params"] = round(model_summary.total_params / 1e6, 2)
    history["total_mult_adds"] = round(model_summary.total_mult_adds / 1e6, 2)

    if include_wandb:
        os.environ["WANDB_MODE"] = "online"
        wandb.login(key=os.getenv("WANDB_API_KEY"))

        wandb.init(
            entity=os.getenv("WANDB_ENTITY_NAME"),
            project=os.getenv("WANDB_PROJECT_NAME"),
            tags=[data_name, model_mode, resnet_mode, "training pipeline"],
            config=hyperparams,
        )

        wandb.log(
            dict(
                model_summary=str(model_summary),
                data_name=data_name,
                num_train_samples=len(train_loader.dataset),
                num_validation_samples=len(validation_loader.dataset),
            )
        )

    optimizer, scheduler = build_optimizer(
        model=model,
        num_epochs=num_epochs,
    )
    criterion = torch.nn.CrossEntropyLoss()

    logger.info(f"Training model with best hyperparameter config:\n{hyperparams}")

    for epoch in range(num_epochs):
        start_time = time.time()
        metrics = train_epoch(
            train_loader=train_loader,
            validation_loader=validation_loader,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=DEVICE,
        )

        history["wall_clock_time"].append(time.time() - start_time)
        history["train_loss"].append(metrics.train_loss)
        history["train_accuracy"].append(metrics.train_acc)
        history["validation_loss"].append(metrics.val_loss)
        history["validation_accuracy"].append(metrics.top1_val_acc)
        history["epochs"].append(epoch + 1)

        if metrics.top1_val_acc > best_top1_val_acc:
            best_top1_val_acc = metrics.top1_val_acc
            best_epoch = epoch + 1
            torch.save(
                obj=model.state_dict(),
                f=f"./models/{ARTIFACT_TAG}_model.pt",
            )  # save best model

        if metrics.top5_val_acc > best_top5_val_acc:
            best_top5_val_acc = metrics.top5_val_acc

        print(
            f"[Epoch: {epoch + 1}/{num_epochs}] [Training: loss={metrics.train_loss:.4f}, acc={metrics.train_acc:.4f}] "
            f"[Validation: loss={metrics.val_loss:.4f}, acc={metrics.top1_val_acc:.4f}] "
            f"[Best: epoch={best_epoch}, top1-acc={best_top1_val_acc:.4f}, top5-acc={best_top5_val_acc:.4f}]"
        )

        if include_wandb:
            wandb.log(
                dict(
                    train_loss=metrics.train_loss,
                    train_accuracy=metrics.train_acc,
                    validation_loss=metrics.val_loss,
                    validation_accuracy=metrics.top1_val_acc,
                    best_top1_val_acc=best_top1_val_acc,
                    best_top5_val_acc=best_top5_val_acc,
                )
            )

    history["accuracy@1"] = best_top1_val_acc
    history["accuracy@5"] = best_top5_val_acc
    history["wct/epoch"] = sum(history["wall_clock_time"]) / num_epochs

    history["semiring"] = semiring
    history["tree_type"] = tree_type
    history["hyperparams"] = hyperparams
    history["num_epochs"] = num_epochs

    dump_data(
        data=history,
        fname=f"{ARTIFACT_TAG}_history.pkl",
        path="./history",
    )

    plot_training_pipeline_history(
        history=history,
        artifact_tag=f"{ARTIFACT_TAG}_history",
    )

    logger.info(
        "Training pipeline pipeline finished successfully. See the ./plots and ./models dirs to see the generated plots and best model respectively"
    )

    return None


def ad_training_pipeline(
    category: str,
    hyperparams: dict,
    semiring: str,
    tree_type: str,
    num_epochs: int,
    ablation: bool,
):
    logger = get_logger(__name__)
    logger.info(
        f"Anomaly detection model training pipeline has started. Category: {category}"
    )

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ARTIFACT_TAG = f"ad_category={category}_ablation={ablation}"

    logger.info("Loading data...")
    loader = load_mvtec(
        batch_size=hyperparams["batch_size"], category=category, include_val=True
    )

    logger.info("Training auto-encoder...")
    fe = build_feature_extractor()
    fe.to(DEVICE)

    if ablation:
        ae = build_ablation_autoencoder(
            in_channels=fe.OUT_CHANNELS,
            latent_dim=hyperparams["latent_dim"],
        )
    else:
        ae = build_autoencoder(
            in_channels=fe.OUT_CHANNELS,
            latent_dim=hyperparams["latent_dim"],
            num_nodes=hyperparams["num_nodes"],
            semiring=semiring,
            tree_type=tree_type,
            seed=hyperparams["seed"],
        )
    ae.to(DEVICE)

    train_loss, val_loss = train_autoencoder(
        ae=ae,
        fe=fe,
        train_loader=loader.train_loader,
        validation_loader=loader.validation_loader,
        num_epochs=num_epochs,
        device=DEVICE,
    )
    torch.save(
        obj=ae.state_dict(),
        f=f"./models/{ARTIFACT_TAG}_ae.pt",
    )

    logger.info("Evaluating anomaly detection on the test set...")
    evaluator = evaluate_anomaly_detector(
        category=category,
        transform=loader.transform,
        img_size=loader.img_size,
        fe=fe,
        ae=ae,
        device=DEVICE,
    )
    print(f"category: {category}, auc roc score: {evaluator.auc_roc_score}")

    history = dict(
        semiring=semiring,
        tree_type=tree_type,
        hyperparams=hyperparams,
        num_epochs=num_epochs,
        y_true=evaluator.y_true,
        y_score=evaluator.y_score,
        auc_roc_score=evaluator.auc_roc_score,
        thresholds=evaluator.thresholds,
        average_precision=evaluator.average_precision,
        fpr=evaluator.fpr,
        tpr=evaluator.tpr,
        precision=evaluator.precision,
        recall=evaluator.recall,
        seg_map=evaluator.seg_map,
        train_loss=train_loss,
        val_loss=val_loss,
    )
    dump_data(
        data=history,
        fname=f"{ARTIFACT_TAG}_history.pkl",
        path="./history",
    )

    logger.info(
        "Anomaly detection training pipeline finished successfully. "
        "Please see the ./models and ./history dirs for the trained model and training history respectifully."
    )

    return None
