import lightning as L
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.utilities.parsing import AttributeDict
from omegaconf import OmegaConf

class TrackNorms(L.Callback):

    # TODO do callbacks happen before or after the method in the main LightningModule?
    # @rank_zero_only # needed?
    def on_after_training_step(self, batch, batch_idx, trainer: L.Trainer, pl_module: L.LightningModule):
        # Log extra metrics
        metrics = {}

        if hasattr(pl_module, "_grad_norms"):
            metrics.update(pl_module._grad_norms)

        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            add_dataloader_idx=False,
            sync_dist=True,
        )


    def on_after_backward(self, trainer: L.Trainer, pl_module: L.LightningModule):
        # example to inspect gradient information in tensorboard
        if OmegaConf.select(trainer.hparams, 'trainer.track_grad_norms'): # TODO dot notation should work with omegaconf?
            norms = {}
            for name, p in pl_module.named_parameters():
                if p.grad is None:
                    continue

                # param_norm = float(p.grad.data.norm(norm_type))
                param_norm = torch.mean(p.grad.data ** 2)
                norms[f"grad_norm.{name}"] = param_norm
            pl_module._grad_norms = norms

