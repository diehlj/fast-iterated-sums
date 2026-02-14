from typing import List

import hydra
import lightning as L
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import grad_norm
from omegaconf import OmegaConf

import src.utils.train
from src.tasks import tasks
from src.utils import registry

log = src.utils.train.get_logger(__name__)
import os

from hydra.utils import get_original_cwd, to_absolute_path

import src.utils as utils
import src.utils.optim_groups
import src.utils.train
import src.utils.optim_groups
from hydra.utils import get_original_cwd, to_absolute_path
import os
import train_helpers

class ImageLightningModule(L.LightningModule):
    def print_optimizer_state(self):
        log.info("Optimizer state_dict: %s", self.optimizers().state_dict())

    def __init__(self, config, num_classes=None):
        super().__init__()
        self.save_hyperparameters(
            config, logger=False
        )  # Now available as self.hparams.
        self.num_classes = num_classes

        self._has_setup = False  # PL has some bugs, so add hooks and make sure they're only called once
        self.setup()

    def setup(self, stage=None):  # XXX stage ever used ?
        # We need to set up the model in setup() because for some reason when training with DDP, one GPU uses much more memory than the others
        # In order to not overwrite the model multiple times during different stages, we need this hack
        # TODO PL 1.5 seems to have an option to skip hooks to avoid this
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/5410#issuecomment-762257024
        if self._has_setup:
            return
        else:
            self._has_setup = True

        # Instantiate model
        self.model = hydra.utils.instantiate(
            self.hparams.model, num_classes=self.num_classes
        )

        self.task = utils.config.instantiate(
            tasks.TASK_REGISTRY, self.hparams.task, model=self.model
        )
        # Extract the modules so they show up in the top level parameter count
        self.loss = self.task.loss
        self.loss_val = self.task.loss
        if hasattr(self.task, "loss_val"):
            self.loss_val = self.task.loss_val
        self.metrics = self.task.metrics

    def on_before_optimizer_step(self, optimizer):
        if (track_grad_norm := self.hparams.trainer.get("track_grad_norm", -1)) != -1:
            self.log_dict(grad_norm(self, norm_type=track_grad_norm))

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, batch_idx, prefix="train"):
        x, y = batch
        y_hat = self.forward(x)

        # Loss
        if prefix == "train":
            loss = self.loss(y_hat, y)
        else:
            loss = self.loss_val(y_hat, y)

        # Metrics
        metrics = self.metrics(y_hat, y)
        metrics["loss"] = loss
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
            sync_dist=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        # Print first batch of training
        if batch_idx == 0 and self.hparams.train.get("print_first_batch", False):
            print(f"\n=== First batch of training - Epoch {self.current_epoch} ===")
            x, y, *z = batch
            print(f"Input shape: {x.shape}, dtype: {x.dtype}")
            print(f"Input element [0,:10,:10]: {x[0, :10, :10]}")
            print(f"Target shape: {y.shape}, dtype: {y.dtype}")
            print(f"Target element [0]: {y[0]}")
            print(f"Input range: [{x.min().item():.4f}, {x.max().item():.4f}]")
            print(f"Target range: [{y.min().item():.4f}, {y.max().item():.4f}]")
            if len(z) > 0:
                print(f"Extra info: {z[0]}")
            print("=" * 50)
        loss = self._shared_step(batch, batch_idx, prefix="train")
        # print(f"Training step {batch_idx}: loss={loss:.2f}")

        # This was used in SequenceModel in combintation with self.track_norms.
        # Log any extra info that the models want to expose (e.g. output norms)
        # module_metrics = {}
        # for module in list(self.modules())[1:]:
        #    if hasattr(module, "metrics"):
        #        module_metrics.update(module.metrics)
        # self.log_dict(
        #    module_metrics,
        #    on_step=True,
        #    on_epoch=False,
        #    prog_bar=False,
        #    add_dataloader_idx=False,
        #    sync_dist=True,
        # )

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if batch_idx == 0 and self.hparams.train.get("print_first_batch", False):
            print(f"\n=== First batch of validation loader {dataloader_idx} ===")
            x, y, *z = batch
            print(f"Input shape: {x.shape}, dtype: {x.dtype}")
            print(f"Input element [0,:10,:10]: {x[0, :10, :10]}")
            print(f"Target shape: {y.shape}, dtype: {y.dtype}")
            print(f"Target element [0]: {y[0]}")
            print(f"Input range: [{x.min().item():.4f}, {x.max().item():.4f}]")
            print(f"Target range: [{y.min().item():.4f}, {y.max().item():.4f}]")
            if len(z) > 0:
                print(f"Extra info: {z[0]}")
            print("=" * 50)

        loss = self._shared_step(
            batch,
            batch_idx,
            prefix="val",  # XXX: self.val_loader_names[dataloader_idx]
        )

        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_step(
            batch,
            batch_idx,
            prefix="test",  # XXX: self.test_loader_names[dataloader_idx]
        )

    def configure_optimizers(self):
        # Set zero weight decay for some params
        if 'optimizer_param_grouping' in self.hparams.train:
            utils.optim_groups.add_optimizer_hooks(self.model, **self.hparams.train.optimizer_param_grouping)

        # Normal parameters
        all_params_named = list(self.named_parameters())
        all_params = [p for n, p in all_params_named]
        params = [p for p in all_params if not hasattr(p, "_optim")]

        optimizer = hydra.utils.instantiate(self.hparams.optimizer, params)
        log.info(f"Optimizer: {optimizer}")

        # Add parameters with special hyperparameters
        hps = [getattr(p, "_optim") for p in all_params if hasattr(p, "_optim")]
        hps = [
            dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
            ]  # Unique dicts
        log.info(f"Hyperparameter groups: {hps}")
        for hp in hps:
            log.info(f"Adding hyperparameter group, hp={hp}")
            params_named = [n for n, p in all_params_named if getattr(p, "_optim", None) == hp]
            params = [p for p in all_params if getattr(p, "_optim", None) == hp]
            # print(f"Adding to hyperparameter group, params_named={params_named}")
            optimizer.add_param_group(
                {"params": params, **self.hparams.optimizer, **hp}
            )
        # print("Parameters in default optimizer groups:")
        # print([n for n, p in all_params_named if not hasattr(p, "_optim")])

        ### Layer Decay ###
        if hasattr(self.hparams.train, 'layer_decay') and self.hparams.train.layer_decay['_name_'] is not None:
            # TODO test
            get_num_layer = utils.config.instantiate(
                registry.layer_decay,
                self.hparams.train.layer_decay['_name_'],
                partial=True,
            )

            # Go through all parameters and get num layer
            layer_wise_groups = {}
            num_max_layers = 0
            for name, p in self.named_parameters():
                # Get layer id for each parameter in the model
                layer_id = get_num_layer(name)

                # Add to layer wise group
                if layer_id not in layer_wise_groups:
                    layer_wise_groups[layer_id] = {
                        'params': [],
                        'lr': None,
                        'weight_decay': self.hparams.optimizer.weight_decay
                    }
                layer_wise_groups[layer_id]['params'].append(p)

                if layer_id > num_max_layers: num_max_layers = layer_id

            # Update lr for each layer
            for layer_id, group in layer_wise_groups.items():
                group['lr'] = self.hparams.optimizer.lr * (
                            self.hparams.train.layer_decay.decay ** (num_max_layers - layer_id))

            # Reset the torch optimizer's param groups
            optimizer.param_groups = []
            for layer_id, group in layer_wise_groups.items():
                log.info(f"Adding layerwise group, layer_id={layer_id}")
                log.info(f"group['lr']={group.get('lr',None)}")
                log.info(f"group['weight_decay']={group.get('weight_decay',None)}")
                log.info(f"length of group['params']={len(group.get('params',None))}")
                optimizer.add_param_group(group)

        # Print optimizer info for debugging
        keys = set([k for hp in hps for k in hp.keys()])  # Special hparams
        utils.train.log_optimizer(log, optimizer, keys)


        # Configure scheduler
        if "scheduler" not in self.hparams:
            return optimizer

        # HACK
        if self.hparams.scheduler["_name_"] not in [
            "cosine_warmup",
            "cosine_warmup_restarts",
        ]:
            if "num_cycles" in self.hparams.scheduler:
                del self.hparams.scheduler["num_cycles"]
        if self.hparams.scheduler["_name_"] not in [
            "linear_warmup",
            "cosine_warmup",
            "cosine_warmup_restarts",
        ]:
            if "num_training_steps" in self.hparams.scheduler:
                del self.hparams.scheduler["num_training_steps"]

        log.info(f"self.hparams.scheduler={self.hparams.scheduler}")
        lr_scheduler = utils.config.instantiate(
            registry.scheduler, self.hparams.scheduler, optimizer
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": self.hparams.train.interval,  # 'epoch' or 'step'
            "monitor": self.hparams.train.monitor,  # e.g. 'val/loss' for ReduceLROnPlateau; XXX not verified
            "name": "trainer/lr",  # default is e.g. 'lr-AdamW'
        }
        log.info(f"Scheduler: {scheduler}")
        return [optimizer], [scheduler]


def create_trainer(config, **kwargs):
    callbacks: List[L.Callback] = []
    logger = None

    # WandB Logging
    if config.get("wandb") is not None:
        # Pass in wandb.init(config=) argument to get the nice 'x.y.0.z' hparams logged
        # Can pass in config_exclude_keys='wandb' to remove certain groups

        # logger = CustomWandbLogger(
        logger = WandbLogger(
            config=utils.config.to_dict(config, recursive=True),
            # settings=wandb.Settings(start_method="fork"),
            **config.wandb,
        )

    # Lightning callbacks
    if "callbacks" in config:
        for _name_, callback_config in config.callbacks.items():
            print(f"Instantiating callback {_name_} with config {callback_config}")
            if config.get("wandb") is None and _name_ in ["learning_rate_monitor"]:
                continue
            if _name_ in ["log_run_info"]:  # See below.
                continue
            log.info(f"Instantiating callback <{registry.callbacks[_name_]}>")
            callback_config._name_ = _name_
            callbacks.append(
                utils.config.instantiate(registry.callbacks, callback_config)
            )

    if (
        config.get("wandb") is not None
        and "callbacks" in config
        and "log_run_info" in config.callbacks
    ):
        from src.callbacks.log_run_info import LogRunInfo

        log.info("Adding LogRunInfo callback for wandb logging")
        callbacks.append(LogRunInfo(config))

    ## Configure ddp automatically
    # if config.trainer.gpus > 1:
    #    print("ddp automatically configured, more than 1 gpu used!")
    #    kwargs["plugins"] = [
    #        L.plugins.DDPPlugin(
    #            find_unused_parameters=True,
    #            gradient_as_bucket_view=False,
    #            # https://pytorch-lightning.readthedocs.io/en/stable/advanced/advanced_gpu.html#ddp-optimizations
    #        )
    #    ]
    #    kwargs["accelerator"] = "ddp"

    ## Add ProgressiveResizing callback
    # if config.callbacks.get("progressive_resizing", None) is not None:
    #    num_stages = len(config.callbacks.progressive_resizing.stage_params)
    #    print(f"Progressive Resizing: {num_stages} stages")
    #    for i, e in enumerate(config.callbacks.progressive_resizing.stage_params):
    #        # Stage params are resolution and epochs, pretty print
    #        print(f"\tStage {i}: {e['resolution']} @ {e['epochs']} epochs")

    # Show progress bar if not using wandb.
    if (
        config.get("wandb", None) is None
    ):  # and config.trainer.get('enable_progress_bar', None) is None:
        config.trainer.enable_progress_bar = True
    else:
        # Less clutter in wandb logs
        callbacks.append( L.pytorch.callbacks.TQDMProgressBar(refresh_rate=10_000, leave=True) )

    kwargs.update(config.trainer)

    # Remove ckpt_path from kwargs (it is only used with `fit`)
    if "ckpt_path" in kwargs:
        del kwargs["ckpt_path"]

    for obsolete in [
        "gpus",
        "weights_summary",
        "progress_bar_refresh_rate",
        "track_grad_norm",
    ]:
        if obsolete in kwargs:
            del kwargs[obsolete]
            print(f"Removed obsolete key '{obsolete}' from trainer kwargs")
    if config.trainer.get("track_grad_norm", -1) != -1:
        # This is now handled in the LightningModule hook on_before_optimizer_step
        log.warning(
            "Using trainer.track_grad_norm (handled in LightningModule hook on_before_optimizer_step). This is very slow!"
        )

    trainer = L.Trainer(logger=logger, callbacks=callbacks, **kwargs)
    return trainer


def train(config):
    if config.train.seed is not None:
        L.seed_everything(config.train.seed, workers=True)
    watch_model = config.trainer.pop("watch_model", False)
    trainer = create_trainer(config)

    # HACK To get the loader config to the dataset
    batch_size = config.loader.batch_size
    num_workers = config.loader.num_workers
    persistent_workers = config.loader.get("persistent_workers", True)
    shuffle = config.loader.get("shuffle", True)
    pin_memory = None
    if "pin_memory" in config.loader:
        pin_memory = config.loader.pin_memory
    dataset = hydra.utils.instantiate(
        config.dataset,
        data_root_dir="./data",  # XXX hardcoded data dir
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        shuffle=shuffle,
    )
    log.info(f"Dataset: {dataset} {type(dataset)}")
    model = ImageLightningModule(config, num_classes=dataset.num_classes)
    model._dataset = dataset  # HACK We need it in log_run_info.py.

    if config.train.get("compile", False):
        # XXX Might have to apply this patch: https://github.com/pytorch/pytorch/issues/136332
        log.info("Compiling the model ..")
        model = torch.compile(model, mode=config.train.get("compile_mode", "default"))

    if (
        not config.train.get("compile", False)
        and config.get("wandb", None) is not None
        and watch_model
    ):
        # XXX For some reason, when calling trainer.logger.watch( model, ..)
        # it does only log the gradients
        # Also, compile and watch seem to not work together.
        log.warning("Using wandb.watch, this is very slow!")
        trainer.logger.watch(model.model, log="all", log_freq=1, log_graph=True)

    if config.train.get("debug", False):
        train_helpers.print_debug_info(model, dataset)
        return
    if config.train.get("debug_show_data", False):
        train_helpers.debug_log_data(dataset, config, trainer.logger)
        return

    if config.trainer.get("ckpt_path", None) is not None:
        log.info(f"Loading checkpoint from {config.trainer.ckpt_path}")
        trainer.fit(model, datamodule=dataset, ckpt_path=config.trainer.ckpt_path)
    else:
        trainer.fit(model, datamodule=dataset)

    wandb.finish()


@hydra.main(version_base="1.3", config_path="configs", config_name="config.yaml")
def main(cfg: OmegaConf):
    print(f"Current working directory : {os.getcwd()}")
    print(f"Orig working directory    : {get_original_cwd()}")
    print(f"to_absolute_path('foo')   : {to_absolute_path('foo')}")
    print(f"to_absolute_path('/foo')  : {to_absolute_path('/foo')}")
    cfg = utils.train.process_config(cfg)  # Resolves, and makes mutable.
    utils.train.print_config(cfg)
    train(cfg)

    from pathlib import Path

    default_data_path = Path(__file__).parent.parent.parent.absolute()
    default_data_path = default_data_path / "data"


if __name__ == "__main__":
    main()