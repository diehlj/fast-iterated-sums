### https://github.com/HazyResearch/transformers/blob/master/src/callbacks/wandb_callbacks.py

import glob
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import wandb
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score

from src.models.sums import pretty_str_tree_one_line
from src.models.fis_resnet_v1 import (
    FISLayerParameterSharingV0,
    FISLayerParameterSharingV1,
    FISLayerParameterSharingV2,
    FISLayerParameterSharingV3,
    FISLayer
)
from src.models.tiny_fashion_net import TinyFashionNetPureFISUltraConfigurable


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    # jd: LoggerCollection is deprecated
    # "directly pass a list of loggers to the Trainer and access the list via the trainer.loggers attribute."
    # https://github.com/Lightning-AI/pytorch-lightning/blob/ce038e876ba013f3ceeda374e45d17f4e8f29d7c/docs/source-pytorch/upgrade/sections/1_7_regular.rst#L10
    #if isinstance(trainer.logger, LoggerCollection):
    #    for logger in trainer.logger:
    #        if isinstance(logger, WandbLogger):
    #            return logger
    if hasattr(trainer, 'loggers') and isinstance(trainer.loggers, list):
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                return logger

    print("[WARN] You are using wandb related callback, but WandbLogger was not found for some reason...")
    return None

class WatchModel(Callback):
    """Make wandb watch model at the beginning of the run."""

    def __init__(self, log: str = "gradients", log_freq: int = 100):
        self.log = log
        self.log_freq = log_freq

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        logger.watch(model=trainer.model, log=self.log, log_freq=self.log_freq)


class UploadCodeAsArtifact(Callback):
    """Upload all *.py files to wandb as an artifact, at the beginning of the run."""

    def __init__(self, code_dir: str):
        self.code_dir = code_dir

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        code = wandb.Artifact("project-source", type="code")
        for path in glob.glob(os.path.join(self.code_dir, "**/*.py"), recursive=True):
            code.add_file(path)

        experiment.log_artifact(code)


class UploadCheckpointsAsArtifact(Callback):
    """Upload checkpoints to wandb as an artifact, at the end of run."""

    def __init__(self, ckpt_dir: str = "checkpoints/", upload_best_only: bool = False):
        self.ckpt_dir = ckpt_dir
        self.upload_best_only = upload_best_only

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in glob.glob(os.path.join(self.ckpt_dir, "**/*.ckpt"), recursive=True):
                ckpts.add_file(path)

        experiment.log_artifact(ckpts)


class LogConfusionMatrix(Callback):
    """Generate confusion matrix every epoch and send it to wandb.
    Expects validation step to return predictions and targets.
    """

    def __init__(self):
        self.preds = []
        self.targets = []
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module) -> None:
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready:
            self.preds.append(outputs["preds"])
            self.targets.append(outputs["targets"])

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate confusion matrix."""
        if self.ready:
            logger = get_wandb_logger(trainer)
            experiment = logger.experiment

            preds = torch.cat(self.preds).cpu().numpy()
            targets = torch.cat(self.targets).cpu().numpy()

            confusion_matrix = metrics.confusion_matrix(y_true=targets, y_pred=preds)

            # set figure size
            plt.figure(figsize=(14, 8))

            # set labels size
            sn.set(font_scale=1.4)

            # set font size
            sn.heatmap(confusion_matrix, annot=True, annot_kws={"size": 8}, fmt="g")

            # names should be uniqe or else charts from different experiments in wandb will overlap
            experiment.log({f"confusion_matrix/{experiment.name}": wandb.Image(plt)}, commit=False)

            # according to wandb docs this should also work but it crashes
            # experiment.log(f{"confusion_matrix/{experiment.name}": plt})

            # reset plot
            plt.clf()

            self.preds.clear()
            self.targets.clear()


class LogF1PrecRecHeatmap(Callback):
    """Generate f1, precision, recall heatmap every epoch and send it to wandb.
    Expects validation step to return predictions and targets.
    """

    def __init__(self, class_names: List[str] = None):
        self.preds = []
        self.targets = []
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready:
            self.preds.append(outputs["preds"])
            self.targets.append(outputs["targets"])

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate f1, precision and recall heatmap."""
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            preds = torch.cat(self.preds).cpu().numpy()
            targets = torch.cat(self.targets).cpu().numpy()
            f1 = f1_score(preds, targets, average=None)
            r = recall_score(preds, targets, average=None)
            p = precision_score(preds, targets, average=None)
            data = [f1, p, r]

            # set figure size
            plt.figure(figsize=(14, 3))

            # set labels size
            sn.set(font_scale=1.2)

            # set font size
            sn.heatmap(
                data,
                annot=True,
                annot_kws={"size": 10},
                fmt=".3f",
                yticklabels=["F1", "Precision", "Recall"],
            )

            # names should be uniqe or else charts from different experiments in wandb will overlap
            experiment.log({f"f1_p_r_heatmap/{experiment.name}": wandb.Image(plt)}, commit=False)

            # reset plot
            plt.clf()

            self.preds.clear()
            self.targets.clear()


class LogImagePredictions(Callback):
    """Logs a validation batch and their predictions to wandb.
    Example adapted from:
        https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY
    """

    def __init__(self, num_samples: int = 8):
        super().__init__()
        self.num_samples = num_samples
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            # get a validation batch from the validation dat loader
            val_samples = next(iter(trainer.datamodule.val_dataloader()))
            val_imgs, val_labels = val_samples

            # run the batch through the network
            val_imgs = val_imgs.to(device=pl_module.device)
            logits = pl_module(val_imgs)
            preds = torch.argmax(logits, axis=-1)

            # log the images as wandb Image
            experiment.log(
                {
                    f"Images/{experiment.name}": [
                        wandb.Image(x, caption=f"Pred:{pred}, Label:{y}")
                        for x, pred, y in zip(
                            val_imgs[: self.num_samples],
                            preds[: self.num_samples],
                            val_labels[: self.num_samples],
                        )
                    ]
                }
            )

class LogDT(Callback):
    """ Log the dt values (from NeurIPS 2021 LSSL submission) """
    def on_train_epoch_end(self, trainer, pl_module):
        log_dict = {}
        for name, m in pl_module.model.named_modules():
            if pl_module.hparams.train.get('log_dt', False) \
                and hasattr(m, "log_dt"):
                log_dict[f"{name}.log_dt"] = (
                    m.log_dt.detach().cpu().numpy().flatten()
                )
                log_dict[f"{name}.log_dt.image"] = wandb.Image(
                    m.log_dt.detach().cpu().numpy().flatten().reshape(1, -1)
                )
                log_dict[f"{name}.log_dt"] = wandb.Table(
                    dataframe=pd.DataFrame(
                        {"log_dt": m.log_dt.detach().cpu().numpy().flatten()}
                    )
                )

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            if trainer.logger is not None:
                trainer.logger.experiment.log(log_dict)

class LogGammaParameters(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        # Handle old models with fis1_gamma and fis2_gamma (backward compatibility)
        if hasattr(pl_module.model, 'fis1_gamma') and hasattr(pl_module.model, 'fis2_gamma'):
            log_dict = {
                "fis1_gamma": pl_module.model.fis1_gamma.item(),
                "fis2_gamma": pl_module.model.fis2_gamma.item(),
            }
            pl_module.log_dict(log_dict, on_epoch=True, sync_dist=True)
        
        # Handle TinyFashionNetPureFISUltraConfigurable with dynamic layer_gammas
        elif isinstance(pl_module.model, TinyFashionNetPureFISUltraConfigurable):
            if hasattr(pl_module.model, 'use_skip_connections') and pl_module.model.use_skip_connections:
                log_dict = {}
                if hasattr(pl_module.model, 'use_gamma_buffer') and pl_module.model.use_gamma_buffer:
                    # Use buffer tensor (shape: (num_layers, 1))
                    if hasattr(pl_module.model, 'layer_gammas_buffer') and pl_module.model.layer_gammas_buffer is not None:
                        for i in range(pl_module.model.layer_gammas_buffer.shape[0]):
                            log_dict[f"layer_{i}_gamma"] = pl_module.model.layer_gammas_buffer[i].item()
                else:
                    # Use ParameterList
                    if hasattr(pl_module.model, 'layer_gammas') and pl_module.model.layer_gammas is not None:
                        for i, gamma in enumerate(pl_module.model.layer_gammas):
                            log_dict[f"layer_{i}_gamma"] = gamma.item()
                
                if log_dict:
                    pl_module.log_dict(log_dict, on_epoch=True, sync_dist=True)

class LogLambdaParameters(Callback):
    """Log Lambda parameters (magnitude and phase) for FIS layers using complex numbers."""
    
    def on_validation_epoch_end(self, trainer, pl_module):
        log_dict = {}
        
        # Find all FIS layers with use_complex=True
        for name, module in pl_module.model.named_modules():
            # Check if module is an FIS layer with complex support
            if isinstance(module, (FISLayerParameterSharingV0, FISLayerParameterSharingV2)):
                if hasattr(module, 'use_complex') and module.use_complex:
                    if hasattr(module, 'discount_nu_log') and hasattr(module, 'discount_theta_log'):
                        # Compute Lambda (same computation as in forward method)
                        Lambda_r = torch.exp(-torch.exp(module.discount_nu_log)) * torch.cos(torch.exp(module.discount_theta_log))
                        Lambda_i = torch.exp(-torch.exp(module.discount_nu_log)) * torch.sin(torch.exp(module.discount_theta_log))
                        
                        # Compute magnitude and phase
                        # Magnitude: |Lambda| = sqrt(Lambda_r^2 + Lambda_i^2)
                        magnitude = torch.sqrt(Lambda_r ** 2 + Lambda_i ** 2)
                        # Phase: arg(Lambda) = atan2(Lambda_i, Lambda_r)
                        phase = torch.atan2(Lambda_i, Lambda_r)
                        
                        # Log per tree
                        # discount_nu_log has shape (1, num_trees, 1, 1)
                        num_trees = magnitude.shape[1]
                        # Sanitize layer name: replace any forward slashes with underscores to avoid breaking WandB grouping
                        layer_name = name if name else "fis_layer"
                        layer_name = layer_name.replace("/", "_")
                        for t in range(num_trees):
                            # Use forward slashes for WandB automatic grouping
                            # Structure: Lambda/{layer_name}/magnitude/tree_{t} and Lambda/{layer_name}/phase/tree_{t}
                            log_dict[f"Lambda/{layer_name}/magnitude/tree_{t}"] = magnitude[0, t, 0, 0].item()
                            log_dict[f"Lambda/{layer_name}/phase/tree_{t}"] = phase[0, t, 0, 0].item()
                        
                        # Log go_to_complex and go_to_real parameters
                        if hasattr(module, 'go_to_complex') and hasattr(module, 'go_to_real'):
                            # go_to_complex: nn.Linear(1, 2) - weight shape (2, 1), bias shape (2,)
                            go_to_complex_weight = module.go_to_complex.weight.data.detach().cpu()
                            go_to_complex_bias = module.go_to_complex.bias.data.detach().cpu()
                            
                            # Log go_to_complex weight components
                            log_dict[f"Lambda/{layer_name}/go_to_complex/weight_0"] = go_to_complex_weight[0, 0].item()
                            log_dict[f"Lambda/{layer_name}/go_to_complex/weight_1"] = go_to_complex_weight[1, 0].item()
                            log_dict[f"Lambda/{layer_name}/go_to_complex/bias_0"] = go_to_complex_bias[0].item()
                            log_dict[f"Lambda/{layer_name}/go_to_complex/bias_1"] = go_to_complex_bias[1].item()
                            
                            # go_to_real: nn.Linear(2, 1) - weight shape (1, 2), bias shape (1,)
                            go_to_real_weight = module.go_to_real.weight.data.detach().cpu()
                            go_to_real_bias = module.go_to_real.bias.data.detach().cpu()
                            
                            # Log go_to_real weight components
                            log_dict[f"Lambda/{layer_name}/go_to_real/weight_0"] = go_to_real_weight[0, 0].item()
                            log_dict[f"Lambda/{layer_name}/go_to_real/weight_1"] = go_to_real_weight[0, 1].item()
                            log_dict[f"Lambda/{layer_name}/go_to_real/bias"] = go_to_real_bias[0].item()
        
        if log_dict:
            pl_module.log_dict(log_dict, on_epoch=True, sync_dist=True)

class LogFISImages(Callback):
    """Log input and output images from FIS layers during validation and at training end."""
    
    def __init__(self, log_freq: int = 1, num_samples: int = 8, log_layers: list | str = None):
        """
        Args:
            log_freq: Log every N validation epochs
            num_samples: Number of sample images to log
            log_layers: List of layer names to log (e.g., ['layer1.fis1', 'layer1.fis2'])
        """
        super().__init__()
        self.log_freq = log_freq
        self.num_samples = num_samples
        self.log_layers = log_layers #or ['layer1.fis1', 'layer2.fis1', 'layer3.fis1']
        self.ready = True
        
    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False
        
    def on_sanity_check_end(self, trainer, pl_module):
        self.ready = True
        
    def _log_fis_images(self, trainer, pl_module, log_prefix=""):
        """Common logging logic for FIS images that can be called during validation or at training end."""
        if self.log_layers is None:
            return
        logger = get_wandb_logger(trainer=trainer)
        if logger is None:
            return
            
        experiment = logger.experiment
        
        # Get validation data
        val_samples = next(iter(trainer.datamodule.val_dataloader()))
        x, y = val_samples
        x = x.to(pl_module.device)
        
        # Log input images with colorbar
        input_images = []
        for i in range(min(self.num_samples, x.shape[0])):
            # Convert to numpy
            img = x[i].cpu().numpy()
            if img.shape[0] == 1:  # Grayscale
                img = img[0]  # Remove channel dimension
                # Create figure with colorbar for grayscale
                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(img, cmap='viridis')
                ax.set_title(f"Input {i}, Label: {y[i].item()}")
                ax.axis('off')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label('Raw Values', rotation=270, labelpad=15)
            else:  # RGB
                img = img.transpose(1, 2, 0)  # CHW -> HWC
                # Normalize to [0, 1] for proper RGB display
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    img_normalized = (img - img_min) / (img_max - img_min)
                else:
                    img_normalized = img
                
                # Create figure for RGB (no colorbar needed for color images)
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.imshow(img_normalized)
                ax.set_title(f"Input {i}, Label: {y[i].item()}\nRange: [{img_min:.2f}, {img_max:.2f}]")
                ax.axis('off')
            input_images.append(wandb.Image(fig, caption=f"Input {i}, Label: {y[i].item()}"))
            plt.close(fig)
        
        # Log input images
        experiment.log({
            f"FIS_Images/{log_prefix}Input_Images": input_images
        }, commit=False)
        
        # Hook into FIS layers to capture projections, outputs, tree info, discount, and alpha parameters
        fis_alpha1 = {}
        fis_alpha2 = {}
        fis_alpha3 = {}
        fis_outputs = {}
        fis_trees = {}
        fis_discounts = {}
        fis_alpha1_params = {}
        fis_alpha2_params = {}
        fis_alpha3_params = {}
        
        # Track which FIS layers we're logging
        detected_fis_layers = []
        
        def hook_fn(name):
            def hook(module, input, output):
                # Always log if this layer was detected (either by name match or auto-detection)
                # Get the input tensor
                x = input[0].detach()
                
                # Compute alpha_1 and alpha_2 projections
                # alpha_1: (num_trees, in_channels) -> project input channels
                # alpha_2: (num_trees, in_channels) -> project input channels
                alpha1_proj = torch.einsum('bchw,tc->bthw', x, module.alpha_1)
                alpha2_proj = torch.einsum('bchw,tc->bthw', x, module.alpha_2)
                
                fis_alpha1[name] = alpha1_proj
                fis_alpha2[name] = alpha2_proj
                fis_outputs[name] = output.detach()
                fis_trees[name] = pretty_str_tree_one_line(module.tree)
                
                # Compute alpha_3 projection if it exists
                if hasattr(module, 'alpha_3') and module.alpha_3 is not None:
                    alpha3_proj = torch.einsum('bchw,tc->bthw', x, module.alpha_3)
                    fis_alpha3[name] = alpha3_proj
                
                # Capture discount parameter if it exists
                #print('hasattr(module, \'discount\')=', hasattr(module, 'discount'))
                #print('module.discount=', module.discount)
                #xx
                if hasattr(module, 'discount') and module.discount is not None:
                    fis_discounts[name] = module.discount.detach()
                
                # Capture alpha parameter values
                fis_alpha1_params[name] = module.alpha_1.detach()
                if hasattr(module, 'alpha_2'):
                    fis_alpha2_params[name] = module.alpha_2.detach()
                if hasattr(module, 'alpha_3') and module.alpha_3 is not None:
                    fis_alpha3_params[name] = module.alpha_3.detach()
            return hook
        
        # Register hooks for FIS layers
        hooks = []
        # FIS layer types to detect
        fis_layer_types = (
            FISLayerParameterSharingV0,
            FISLayerParameterSharingV1,
            FISLayerParameterSharingV2,
            FISLayerParameterSharingV3,
            FISLayer
        )
        
        # print('trying to register hooks for FIS layers; log_layers=', self.log_layers)
        for name, module in pl_module.model.named_modules():
            # Check if this is an FIS layer
            is_fis_layer = isinstance(module, fis_layer_types)
            
            # If log_layers is 'all', auto-detect all FIS layers by type
            # Otherwise, require name matching
            should_log = False
            if is_fis_layer:
                if self.log_layers == 'all':
                    should_log = True
                elif any(layer in name for layer in self.log_layers):
                    should_log = True
            
            if should_log:
                detected_fis_layers.append(name)
                # print('registering hook for', name)
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        # Forward pass to capture FIS outputs
        with torch.no_grad():
            _ = pl_module.model(x)
        
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Log comparison images (alpha_1, alpha_2, alpha_3 if exists, output) with colorbar
        if fis_alpha1 and fis_alpha2 and fis_outputs:
            for layer_name in fis_alpha1.keys():
                if layer_name in fis_alpha2 and layer_name in fis_outputs:
                    comparison_images = []
                    has_alpha3 = layer_name in fis_alpha3
                    
                    for i in range(min(self.num_samples, fis_alpha1[layer_name].shape[0])):
                        # Get first tree projection for alpha_1 and alpha_2
                        alpha1_img = fis_alpha1[layer_name][i, 0].cpu().numpy()  # First tree
                        alpha2_img = fis_alpha2[layer_name][i, 0].cpu().numpy()  # First tree
                        output_img = fis_outputs[layer_name][i, 0].cpu().numpy()  # First channel
                        #if i == 0 and layer_name == 'layer1.fis1':
                        #    print("alpha1_img=\n", fis_alpha1[layer_name][i, 0])
                        #    print("alpha2_img=\n", fis_alpha2[layer_name][i, 0])
                        #    print("output_img=\n", fis_outputs[layer_name][i, 0])
                        
                        if has_alpha3:
                            # Create four-panel comparison with colorbar - let matplotlib handle normalization
                            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 5))
                            
                            # Get alpha_3 projection
                            alpha3_img = fis_alpha3[layer_name][i, 0].cpu().numpy()  # First tree
                            
                            # Alpha_1 projection
                            im1 = ax1.imshow(alpha1_img, cmap='viridis')
                            ax1.set_title(f"{layer_name} α₁ {i}")
                            ax1.axis('off')
                            
                            # Alpha_2 projection
                            im2 = ax2.imshow(alpha2_img, cmap='viridis')
                            ax2.set_title(f"{layer_name} α₂ {i}")
                            ax2.axis('off')
                            
                            # Alpha_3 projection
                            im3 = ax3.imshow(alpha3_img, cmap='viridis')
                            ax3.set_title(f"{layer_name} α₃ {i}")
                            ax3.axis('off')
                            
                            # FIS output
                            im4 = ax4.imshow(output_img, cmap='viridis')
                            ax4.set_title(f"{layer_name} Output {i}")
                            ax4.axis('off')
                            
                            # Add colorbar for all images
                            cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
                            cbar1.set_label('Raw Values', rotation=270, labelpad=15)
                            cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
                            cbar2.set_label('Raw Values', rotation=270, labelpad=15)
                            cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
                            cbar3.set_label('Raw Values', rotation=270, labelpad=15)
                            cbar4 = plt.colorbar(im4, ax=ax4, shrink=0.8)
                            cbar4.set_label('Raw Values', rotation=270, labelpad=15)
                            
                            plt.tight_layout()
                            tree_str = fis_trees[layer_name]
                            comparison_images.append(wandb.Image(fig, caption=f"{layer_name} α₁ vs α₂ vs α₃ vs Output {i}\nTree: {tree_str}"))
                            plt.close(fig)
                        else:
                            # Create three-panel comparison with colorbar - let matplotlib handle normalization
                            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
                            
                            # Alpha_1 projection
                            im1 = ax1.imshow(alpha1_img, cmap='viridis')
                            ax1.set_title(f"{layer_name} α₁ {i}")
                            ax1.axis('off')
                            
                            # Alpha_2 projection
                            im2 = ax2.imshow(alpha2_img, cmap='viridis')
                            ax2.set_title(f"{layer_name} α₂ {i}")
                            ax2.axis('off')
                            
                            # FIS output
                            im3 = ax3.imshow(output_img, cmap='viridis')
                            ax3.set_title(f"{layer_name} Output {i}")
                            ax3.axis('off')
                            
                            # Add colorbar for all images
                            cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
                            cbar1.set_label('Raw Values', rotation=270, labelpad=15)
                            cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
                            cbar2.set_label('Raw Values', rotation=270, labelpad=15)
                            cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
                            cbar3.set_label('Raw Values', rotation=270, labelpad=15)
                            
                            plt.tight_layout()
                            tree_str = fis_trees[layer_name]
                            comparison_images.append(wandb.Image(fig, caption=f"{layer_name} α₁ vs α₂ vs Output {i}\nTree: {tree_str}"))
                            plt.close(fig)
                    
                    experiment.log({
                        f"FIS_Images/{log_prefix}{layer_name}_Projections_vs_Output": comparison_images
                    }, commit=False)
        
        # Log discount parameter images
        if fis_discounts:
            for layer_name, discount in fis_discounts.items():
                # discount shape: (1, in_channels, 1, 1)
                discount_np = discount.cpu().numpy()
                # Squeeze to (in_channels,)
                discount_vals = discount_np.squeeze()
                
                # Create heatmap of discount values
                fig, ax = plt.subplots(figsize=(10, 6))
                im = ax.imshow(discount_vals.reshape(1, -1), cmap='viridis', aspect='auto')
                ax.set_title(f"{layer_name} Discount Parameter")
                ax.set_xlabel('Channel')
                ax.set_ylabel('')
                ax.set_yticks([])
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Discount Value', rotation=270, labelpad=15)
                
                # Add text annotations for each channel
                for i, val in enumerate(discount_vals):
                    ax.text(i, 0, f'{val:.4f}', ha='center', va='center', 
                           color='white' if val < discount_vals.mean() else 'black', fontsize=8)
                
                plt.tight_layout()
                experiment.log({
                    f"FIS_Images/{log_prefix}{layer_name}_Discount": wandb.Image(fig, caption=f"{layer_name} Discount Parameter")
                }, commit=False)
                plt.close(fig)
        
        # Log alpha parameter images
        if fis_alpha1_params:
            for layer_name in fis_alpha1_params.keys():
                alpha1_param = fis_alpha1_params[layer_name]  # shape: (num_trees, in_channels)
                alpha1_np = alpha1_param.cpu().numpy()
                
                # Create heatmap of alpha_1 parameter values
                fig, ax = plt.subplots(figsize=(12, 8))
                im = ax.imshow(alpha1_np, cmap='viridis', aspect='auto')
                ax.set_title(f"{layer_name} α₁ Parameter Values")
                ax.set_xlabel('Channel')
                ax.set_ylabel('Tree')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('α₁ Value', rotation=270, labelpad=15)
                
                plt.tight_layout()
                experiment.log({
                    f"FIS_Images/{log_prefix}{layer_name}_Alpha1_Params": wandb.Image(fig, caption=f"{layer_name} α₁ Parameter Values")
                }, commit=False)
                plt.close(fig)
                
                # Log alpha_2 if it exists
                if layer_name in fis_alpha2_params:
                    alpha2_param = fis_alpha2_params[layer_name]  # shape: (num_trees, in_channels)
                    alpha2_np = alpha2_param.cpu().numpy()
                    
                    # Create heatmap of alpha_2 parameter values
                    fig, ax = plt.subplots(figsize=(12, 8))
                    im = ax.imshow(alpha2_np, cmap='viridis', aspect='auto')
                    ax.set_title(f"{layer_name} α₂ Parameter Values")
                    ax.set_xlabel('Channel')
                    ax.set_ylabel('Tree')
                    
                    # Add colorbar
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label('α₂ Value', rotation=270, labelpad=15)
                    
                    plt.tight_layout()
                    experiment.log({
                        f"FIS_Images/{log_prefix}{layer_name}_Alpha2_Params": wandb.Image(fig, caption=f"{layer_name} α₂ Parameter Values")
                    }, commit=False)
                    plt.close(fig)
                
                # Log alpha_3 if it exists
                if layer_name in fis_alpha3_params:
                    alpha3_param = fis_alpha3_params[layer_name]  # shape: (num_trees, in_channels) or (num_trees // 2, in_channels)
                    alpha3_np = alpha3_param.cpu().numpy()
                    
                    # Create heatmap of alpha_3 parameter values
                    fig, ax = plt.subplots(figsize=(12, 8))
                    im = ax.imshow(alpha3_np, cmap='viridis', aspect='auto')
                    ax.set_title(f"{layer_name} α₃ Parameter Values")
                    ax.set_xlabel('Channel')
                    ax.set_ylabel('Tree')
                    
                    # Add colorbar
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label('α₃ Value', rotation=270, labelpad=15)
                    
                    plt.tight_layout()
                    experiment.log({
                        f"FIS_Images/{log_prefix}{layer_name}_Alpha3_Params": wandb.Image(fig, caption=f"{layer_name} α₃ Parameter Values")
                    }, commit=False)
                    plt.close(fig)
        
    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.ready or trainer.current_epoch % self.log_freq != 0:
            return
        self._log_fis_images(trainer, pl_module)
        
    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        """Perform final logging of FIS images after training is complete."""
        if self.ready:
            self._log_fis_images(trainer, pl_module, log_prefix="") #log_prefix="Final_")


class LogCNNImages(Callback):
    """Log input and output images from CNN layers during validation and at training end.
    
    This callback visualizes how CNN layers transform multi-dimensional feature maps.
    For each CNN layer, it shows side-by-side comparisons of input vs output.
    
    Important: CNN layers have multi-dimensional inputs/outputs of shape 
    (batch_size, channels, height, width). This callback only visualizes the 
    FIRST CHANNEL (channel 0) of these feature maps.
    
    What you see in WandB:
    - Input images: Original validation data
    - {layer_name}_Input_vs_Output: Side-by-side comparison showing:
      * Left: Channel 0 of the input feature map to the layer
      * Right: Channel 0 of the output feature map from the layer
      * Colorbars show the actual raw activation values
    
    Note: You're seeing only one slice (channel 0) of the full multi-dimensional 
    feature space. For convolutions that are _not_ depthwise, you're hence not
    seeing the complete input.
    
    Example usage:
        cnn_callback = LogCNNImages(
            log_freq=1,  # Log every epoch
            num_samples=8,  # Number of samples to visualize
            log_layers=['stem', 'block1b', 'block2a']  # Specific layers to log
        )
    """
    
    def __init__(self, log_freq: int = 1, num_samples: int = 8, log_layers: list = None):
        """
        Args:
            log_freq: Log every N validation epochs
            num_samples: Number of sample images to log
            log_layers: List of layer names to log (e.g., ['stem', 'block1b', 'block2a'])
        """
        super().__init__()
        self.log_freq = log_freq
        self.num_samples = num_samples
        self.log_layers = log_layers or ['stem', 'block1b', 'block2a', 'block2b', 'block3a']
        self.ready = True
        
    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False
        
    def on_sanity_check_end(self, trainer, pl_module):
        self.ready = True
        
    def _log_cnn_images(self, trainer, pl_module, log_prefix=""):
        """Common logging logic for CNN images that can be called during validation or at training end."""
        logger = get_wandb_logger(trainer=trainer)
        if logger is None:
            return
            
        experiment = logger.experiment
        
        # Get validation data
        val_samples = next(iter(trainer.datamodule.val_dataloader()))
        x, y = val_samples
        x = x.to(pl_module.device)
        
        # Log input images with colorbar
        input_images = []
        for i in range(min(self.num_samples, x.shape[0])):
            # Convert to numpy
            img = x[i].cpu().numpy()
            if img.shape[0] == 1:  # Grayscale
                img = img[0]  # Remove channel dimension
            else:  # RGB
                img = img.transpose(1, 2, 0)  # CHW -> HWC
            
            # Create figure with colorbar - let matplotlib handle normalization
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(img, cmap='viridis')
            ax.set_title(f"Input {i}, Label: {y[i].item()}")
            ax.axis('off')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Raw Values', rotation=270, labelpad=15)
            
            # Convert to wandb image
            input_images.append(wandb.Image(fig, caption=f"Input {i}, Label: {y[i].item()}"))
            plt.close(fig)
        
        # Log input images
        experiment.log({
            f"CNN_Images/{log_prefix}Input_Images": input_images
        }, commit=False)
        
        # Hook into CNN layers to capture inputs and outputs
        cnn_inputs = {}
        cnn_outputs = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if name in self.log_layers:
                    cnn_inputs[name] = input[0].detach()  # First input tensor
                    cnn_outputs[name] = output.detach()
            return hook
        
        # Register hooks for CNN layers
        hooks = []
        for name, module in pl_module.model.named_modules():
            if any(layer in name for layer in self.log_layers):
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        # Forward pass to capture CNN outputs
        with torch.no_grad():
            _ = pl_module.model(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Log comparison images (CNN layer input vs output) with colorbar
        if cnn_inputs and cnn_outputs:
            for layer_name in cnn_inputs.keys():
                if layer_name in cnn_outputs:
                    comparison_images = []
                    for i in range(min(self.num_samples, cnn_inputs[layer_name].shape[0])):
                        # Get first channel for visualization
                        input_img = cnn_inputs[layer_name][i, 0].cpu().numpy()
                        output_img = cnn_outputs[layer_name][i, 0].cpu().numpy()
                        
                        # Create side-by-side comparison with colorbar - let matplotlib handle normalization
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                        
                        # CNN input image
                        im1 = ax1.imshow(input_img, cmap='viridis')
                        ax1.set_title(f"{layer_name} Input {i}")
                        ax1.axis('off')
                        
                        # CNN output image
                        im2 = ax2.imshow(output_img, cmap='viridis')
                        ax2.set_title(f"{layer_name} Output {i}")
                        ax2.axis('off')
                        
                        # Add colorbar for both images
                        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
                        cbar1.set_label('Raw Values', rotation=270, labelpad=15)
                        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
                        cbar2.set_label('Raw Values', rotation=270, labelpad=15)
                        
                        plt.tight_layout()
                        comparison_images.append(wandb.Image(fig, caption=f"{layer_name} Input vs Output {i}"))
                        plt.close(fig)
                    
                    experiment.log({
                        f"CNN_Images/{log_prefix}{layer_name}_Input_vs_Output": comparison_images
                    }, commit=False)
        
    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.ready or trainer.current_epoch % self.log_freq != 0:
            return
        self._log_cnn_images(trainer, pl_module)
        
    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        """Perform final logging of CNN images after training is complete."""
        if self.ready:
            self._log_cnn_images(trainer, pl_module, log_prefix="") #log_prefix="Final_")


class LogPositionalEncoding(Callback):
    """Log images of sinusoidal positional encodings during validation and at training end.
    
    This callback visualizes the pre-computed positional encodings from PE modules.
    It shows different channels of the encoding as 2D images, allowing you to see
    the spatial patterns learned by the positional encoding.
    
    What you see in WandB:
    - {module_name}_PE_Channel_{i}: Individual channel visualizations
    - {module_name}_PE_Channel_Grid: Grid view of multiple channels
    - {module_name}_PE_Stats: Summary statistics about the encoding
    
    Example usage:
        pe_callback = LogPositionalEncoding(
            log_freq=1,  # Log every epoch
            num_channels=16,  # Number of channels to visualize
            log_modules=['pe', 'pos_enc']  # Module names containing 'pe' or 'pos_enc'
        )
    """
    
    def __init__(self, log_freq: int = 1, num_channels: int = 16, log_modules: list = None):
        """
        Args:
            log_freq: Log every N validation epochs
            num_channels: Number of channels to visualize (will show first N channels)
            log_modules: List of module name patterns to search for (e.g., ['pe', 'pos_enc'])
        """
        super().__init__()
        self.log_freq = log_freq
        self.num_channels = num_channels
        self.log_modules = log_modules or ['pe', 'pos_enc', 'positional']
        self.ready = True
        
    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False
        
    def on_sanity_check_end(self, trainer, pl_module):
        self.ready = True
        
    def _is_sinusoidal_pe(self, module):
        """Check if module is a sinusoidal positional encoding."""
        if not hasattr(module, 'pe_type'):
            return False
        sinusoidal_types = [
            'sinusoidal_2d_geometric_prog',
            'sinusoidal_2d_linear_prog',
            'sinusoidal_2d',  # backward compat
            'fourier'  # backward compat (alias for linear)
        ]
        return module.pe_type in sinusoidal_types
    
    def _log_pe_images(self, trainer, pl_module, log_prefix=""):
        """Common logging logic for positional encoding images."""
        logger = get_wandb_logger(trainer=trainer)
        if logger is None:
            return
            
        experiment = logger.experiment
        
        # Find all PE modules with sinusoidal encodings
        pe_modules = {}
        for name, module in pl_module.model.named_modules():
            # Check if module name matches any pattern
            if any(pattern.lower() in name.lower() for pattern in self.log_modules):
                if self._is_sinusoidal_pe(module):
                    if hasattr(module, 'pe_encoding'):
                        pe_modules[name] = module
        
        if not pe_modules:
            return
        
        # Log positional encodings for each module
        for module_name, module in pe_modules.items():
            pe_encoding = module.pe_encoding  # (1, dim, H, W)
            pe_np = pe_encoding.squeeze(0).cpu().numpy()  # (dim, H, W)
            
            dim, H, W = pe_np.shape
            num_channels_to_show = min(self.num_channels, dim)
            
            # Log individual channel images
            channel_images = []
            for ch_idx in range(num_channels_to_show):
                channel_data = pe_np[ch_idx]  # (H, W)
                
                fig, ax = plt.subplots(figsize=(8, 7))
                im = ax.imshow(channel_data, cmap='RdBu', aspect='auto')
                ax.set_title(f"{module_name} PE Channel {ch_idx}\nType: {module.pe_type}")
                ax.set_xlabel('Width')
                ax.set_ylabel('Height')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Encoding Value', rotation=270, labelpad=15)
                
                plt.tight_layout()
                channel_images.append(wandb.Image(fig, caption=f"{module_name} PE Channel {ch_idx}"))
                plt.close(fig)
            
            experiment.log({
                f"PositionalEncoding/{log_prefix}{module_name}_Channels": channel_images
            }, commit=False)
            
            # Log grid view of multiple channels
            if num_channels_to_show > 0:
                # Create a grid of channels
                grid_size = int(np.ceil(np.sqrt(num_channels_to_show)))
                fig, axes = plt.subplots(grid_size, grid_size, figsize=(16, 16))
                axes = axes.flatten() if num_channels_to_show > 1 else [axes]
                
                for idx in range(grid_size * grid_size):
                    ax = axes[idx]
                    if idx < num_channels_to_show:
                        channel_data = pe_np[idx]
                        im = ax.imshow(channel_data, cmap='RdBu', aspect='auto')
                        ax.set_title(f"Ch {idx}")
                        ax.axis('off')
                        plt.colorbar(im, ax=ax, shrink=0.8)
                    else:
                        ax.axis('off')
                
                plt.suptitle(f"{module_name} Positional Encoding - Channel Grid\nType: {module.pe_type}", 
                           fontsize=14, y=0.995)
                plt.tight_layout()
                experiment.log({
                    f"PositionalEncoding/{log_prefix}{module_name}_Channel_Grid": wandb.Image(fig)
                }, commit=False)
                plt.close(fig)
            
            # Log statistics
            stats = {
                f"PositionalEncoding/{log_prefix}{module_name}_mean": float(pe_np.mean()),
                f"PositionalEncoding/{log_prefix}{module_name}_std": float(pe_np.std()),
                f"PositionalEncoding/{log_prefix}{module_name}_min": float(pe_np.min()),
                f"PositionalEncoding/{log_prefix}{module_name}_max": float(pe_np.max()),
                f"PositionalEncoding/{log_prefix}{module_name}_shape": f"{dim}x{H}x{W}",
                f"PositionalEncoding/{log_prefix}{module_name}_type": module.pe_type,
            }
            
            # Add module-specific parameters
            if hasattr(module, 'base'):
                stats[f"PositionalEncoding/{log_prefix}{module_name}_base"] = module.base
            if hasattr(module, 'num_bands'):
                stats[f"PositionalEncoding/{log_prefix}{module_name}_num_bands"] = module.num_bands
            if hasattr(module, 'max_freq'):
                stats[f"PositionalEncoding/{log_prefix}{module_name}_max_freq"] = module.max_freq
            
            experiment.log(stats, commit=False)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.ready or trainer.current_epoch % self.log_freq != 0:
            return
        self._log_pe_images(trainer, pl_module)
        
    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        """Perform final logging of positional encodings after training is complete."""
        if self.ready:
            self._log_pe_images(trainer, pl_module, log_prefix="Final_")
