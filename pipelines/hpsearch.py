import os

import torch
import wandb
from dotenv import load_dotenv

from shared.helpers import dump_data, get_logger
from steps import build_model, build_optimizer, load_data, train_epoch

load_dotenv()


def hpsearch_pipeline(
    data_name: str,
    model_mode: str,
    resnet_mode: str,
    sweep_config: dict[str, dict | str],
) -> None:
    logger = get_logger(__name__)

    logger.info("Hyperparameter tuning pipeline has started.")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.environ["WANDB_MODE"] = "online"
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        entity=os.getenv("WANDB_ENTITY_NAME"),
        project=os.getenv("WANDB_PROJECT_NAME"),
    )

    def search_hp(config=None):
        with wandb.init(
            config=config,
            mode="online",
            tags=[data_name, model_mode, resnet_mode, "hpsearch pipeline"],
        ):
            config = wandb.config

            best_val_acc = 0.0

            train_loader, validation_loader = load_data(
                data_name=data_name, batch_size=config.batch_size
            )

            model = build_model(
                model_mode=model_mode,
                data_name=data_name,
                resnet_mode=resnet_mode,
                pool_mode=config.pool_mode,
                num_nodes=config.num_nodes,
                seed=config.seed,
                semiring=config.semiring,
            )
            model.to(device=DEVICE)

            optimizer, scheduler = build_optimizer(
                model=model, num_epochs=config.num_epochs
            )

            criterion = torch.nn.CrossEntropyLoss()

            for _ in range(config.num_epochs):
                metrics = train_epoch(
                    train_loader=train_loader,
                    validation_loader=validation_loader,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    criterion=criterion,
                    device=DEVICE,
                )

                if metrics.top1_val_acc > best_val_acc:
                    best_val_acc = metrics.top1_val_acc

                wandb.log(
                    dict(
                        train_loss=metrics.train_loss,
                        train_accuracy=metrics.train_acc,
                        validation_loss=metrics.val_loss,
                        validation_accuracy=metrics.top1_val_acc,
                    )
                )

            wandb.log({"best_validation_accuracy": best_val_acc})

    wandb.agent(sweep_id=sweep_id, function=search_hp)

    # get best run config
    logger.info("Getting best run...")
    api = wandb.Api()
    sweep = api.sweep(
        f"{os.getenv('WANDB_ENTITY_NAME')}/{os.getenv('WANDB_PROJECT_NAME')}/{sweep_id}"
    )
    best_run = sweep.best_run()
    best_run_info = {
        "data_name": data_name,
        "model_mode": model_mode,
        "resnet_mode": resnet_mode,
        "id": best_run.id,
        "name": best_run.name,
        "metric": best_run.summary.get("best_validation_accuracy"),
        "config": best_run.config,
    }
    print(f"Best run info:\n{best_run_info}")
    dump_data(
        data=best_run_info,
        fname=f"data_name={data_name}_resnet_mode={resnet_mode}_model_mode={model_mode}_best_run_config.pkl",
        path="./hpruns",
    )
    logger.info("Best run info written to ./hpruns")

    logger.info("Hyperparameter search pipeline finished successfully.")

    return None
