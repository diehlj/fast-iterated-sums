import os
import sys
import hydra
import wandb
import subprocess
import torchinfo
from lightning.pytorch import Callback
from lightning.pytorch.utilities import rank_zero_only
from .wandb import get_wandb_logger

from hydra.core.hydra_config import HydraConfig
# from pprint import pprint
from pprint import pformat
from omegaconf import OmegaConf

class LogRunInfo(Callback):
    """Log run information to wandb including config, directories, and model summaries."""

    def __init__(self, config):
        """
        Args:
            config: The full hydra config object
        """
        self.config = config
        hydra_cfg = HydraConfig.get()
        self.hydra_choices = OmegaConf.to_container(hydra_cfg.runtime.choices)
        # Capture command line early before it might be modified
        self.command_line = " ".join(sys.argv)

    @rank_zero_only
    def on_fit_start(self, trainer, pl_module):
        """Log run information at the start of training."""
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        # Log command line to wandb.config so it shows up in the UI
        # This will override the service command that wandb logs by default
        experiment.config.update({"command": self.command_line}, allow_val_change=True)

        # Collect all run information into a single table
        run_info_data = []

        job_id = os.environ.get('SLURM_JOB_ID', None)
        if job_id is not None:
            run_info_data.append(["Job ID", job_id])

        # Command line:
        run_info_data.append(["Command Line", self.command_line])

        # git diff:
        git_diff = self._get_git_diff()
        run_info_data.append(["Git Diff", git_diff])

        # hydra choices:
        run_info_data.append(["Hydra Choices", pformat(self.hydra_choices)])

        # 1. Config tree
        config_tree_path = "config_tree.txt"
        if os.path.exists(config_tree_path):
            with open(config_tree_path, "r") as f:
                config_tree_content = f.read()
            run_info_data.append(["Config Tree", config_tree_content])
        else:
            run_info_data.append(["Config Tree", "config_tree.txt not found"])

        # 2. Working directory
        working_dir = os.getcwd()
        run_info_data.append(["Working Directory", working_dir])

        # 3. Output directory
        try:
            output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            run_info_data.append(["Output Directory", output_dir])
        except Exception as e:
            run_info_data.append(["Output Directory", f"Error: {str(e)}"])

        # 4. Model summaries
        model_info = self._get_model_summaries(pl_module)
        run_info_data.extend(model_info)

        # Log everything as a single table
        run_info_table = wandb.Table(
            columns=["Component", "Information"], 
            data=run_info_data
        )
        experiment.log({"run_info": run_info_table})

    def _get_git_diff(self):
        """Get git diff as a string."""
        try:
            # Get git diff of staged and unstaged changes
            result = subprocess.run(
                ['git', 'diff', 'HEAD'], 
                capture_output=True, 
                text=True, 
                cwd=os.getcwd(),
                timeout=30
            )
            
            if result.returncode == 0:
                git_diff = result.stdout
                if not git_diff.strip():
                    # No changes, get last commit info instead
                    commit_result = subprocess.run(
                        ['git', 'log', '-1', '--oneline'], 
                        capture_output=True, 
                        text=True, 
                        cwd=os.getcwd(),
                        timeout=10
                    )
                    return f"No changes. Last commit: {commit_result.stdout.strip()}"
                return git_diff
            else:
                return f"Git diff error: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return "Git diff timeout"
        except FileNotFoundError:
            return "Git not found"
        except Exception as e:
            return f"Git diff error: {str(e)}"

    def _get_model_summaries(self, pl_module):
        """Generate model summaries and return as list of [component, info] pairs."""
        try:
            # Make sure the model is set up
            if not hasattr(pl_module, 'model') or pl_module.model is None:
                return [["Model Summaries", "Model not yet initialized"]]

            # Get a sample batch for sizing
            train_dataloader = pl_module._dataset.train_dataloader()
            x, _ = next(iter(train_dataloader))

            # Move to appropriate device if needed
            device = next(pl_module.parameters()).device
            if hasattr(x, 'to'):
                x = x.to(device)

            # Model summary  
            model_input_size = x.shape
            model_summary = torchinfo.summary(
                pl_module.model, 
                verbose=0, 
                input_size=model_input_size,
                device=device,
                depth=5,
                col_names=["input_size", "output_size", "num_params", "kernel_size", "groups", "mult_adds"],
            )

            # Return all model information as list of [component, info] pairs
            model_info = [
                ["Model Input Size", str(model_input_size)],
                ["Model Summary", str(model_summary)],
                ["Model Structure", str(pl_module.model)],
                ["Task", str(pl_module.task)],
                ["Task Loss", str(pl_module.task.loss)],
                ["Task Metrics", str(pl_module.task.metrics)],
                ["Dataset", str(pl_module._dataset)]
            ]
            
            # Add loss_val if it exists
            if hasattr(pl_module.task, 'loss_val'):
                model_info.append(["Task Loss Val", str(pl_module.task.loss_val)])
            
            return model_info

        except Exception as e:
            return [["Model Summaries Error", str(e)]]
