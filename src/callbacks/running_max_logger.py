from lightning.pytorch.callbacks import Callback
import torch


class RunningMaxLogger(Callback):
    """Callback to log running max of validation metrics to wandb"""
    
    def __init__(self, metrics_to_track=None):
        super().__init__()
        # Default to tracking accuracy metrics if none specified
        if metrics_to_track is None:
            self.metrics_to_track = ['val/accuracy', 'val/acc']
        else:
            self.metrics_to_track = metrics_to_track
        
        # Store running max for each metric
        self.running_max = {}
        
    def on_validation_epoch_end(self, trainer, pl_module):
        """Log running max after each validation epoch"""
        if not trainer.is_global_zero:
            return
            
        # Get current metrics from trainer
        current_metrics = trainer.callback_metrics
        
        for metric_name in self.metrics_to_track:
            if metric_name in current_metrics:
                current_value = current_metrics[metric_name]
                
                # Initialize if first time seeing this metric
                if metric_name not in self.running_max:
                    self.running_max[metric_name] = current_value
                else:
                    # Update running max
                    self.running_max[metric_name] = torch.max(
                        self.running_max[metric_name], 
                        current_value
                    )
                
                # Log to wandb
                if hasattr(trainer, 'logger') and trainer.logger is not None \
                   and hasattr(trainer.logger, 'experiment') \
                   and hasattr(trainer.logger.experiment, 'log'):
                    trainer.logger.experiment.log({
                        f"{metric_name}_running_max": self.running_max[metric_name],
                        # f"{metric_name}_current": current_value
                    })
