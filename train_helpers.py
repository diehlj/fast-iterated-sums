import torchinfo
import torch
import pytorch_lightning as L
import numpy as np
import matplotlib.pyplot as plt
import wandb
import src.utils as utils

def print_debug_info(model, dataset):
    dataset.setup('train')
    print(f"\n\n################ DEBUG MODE ###############")

    print(f"\n\n################ train data size #####")
    print(f"Dataset size: {len(dataset.train_dataloader().dataset)}")

    print(f"\n\n################ val data size #####")
    val_dataloader = dataset.val_dataloader()
    if isinstance(val_dataloader, list):
        for dl in val_dataloader:
            print(f"Dataset size: {len(dl.dataset)}")
    else:
        print(f"Dataset size: {len(val_dataloader.dataset)}")
    print(f"\n\n################ test data size #####")
    test_dataloader = dataset.test_dataloader()
    if isinstance(test_dataloader, list):
        for dl in test_dataloader:
            print(f"Dataset size: {len(dl.dataset)}")
    else:
        print(f"Dataset size: {len(test_dataloader.dataset)}")

    x, y = next(iter(dataset.train_dataloader()))
    print('x.shape=', x.shape, 'y.shape=', y.shape)
    model_input_size = x.shape

    print(f"\n\n############### model: ################\ninput size: {model_input_size}")
    try:
        model_summary = torchinfo.summary(model.model, verbose=0, input_size=model_input_size, depth=7,
                                        col_names=["input_size", "output_size", "num_params", "kernel_size", "groups", "mult_adds"],)
    except Exception as e:
        # 'groups' not supported in older torchinfo versions
        model_summary = torchinfo.summary(model.model, verbose=0, input_size=model_input_size, depth=5,
                                        col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],)
    ## draw_graph(model.model, depth=graph_depth, expand_nested=expand_nested, graph_name='model', input_size=model_input_size, save_graph=True)

    # loop over ALL named parameters and print their names and shapes
    print("\nAll model parameters:")
    for name, param in model.model.named_parameters():
        print(f"{name}: {param.shape}, requires_grad={param.requires_grad}")

    print("\n"+str(model_summary))
    print("\n"+str(model.model))

    # check if 'batchnorm' is in str(model.model).lower():
    if any('batchnorm' in str(type(m)).lower() for m in model.model.modules()):
        print("\nModel contains BatchNorm layers. Skipping check for gradient cross-contamination.")
    else:
        model.to('cuda')
        x = x.to('cuda')
        x.requires_grad_(True)

        y_hat = model.model(x)
        fake_loss = torch.sum( y_hat[0]**2 )
        print('y_hat.shape=', y_hat.shape)
        fake_loss.backward(retain_graph=True)

        # Check for cross-contamination:
        assert not torch.allclose(x.grad[0], torch.zeros_like(x.grad[0]), rtol=1e-3, atol=1e-5)
        torch.testing.assert_close(x.grad[1], torch.zeros_like(x.grad[0]), rtol=1e-3, atol=1e-5)

def debug_log_data(dataset, config, logger):
    """
    Debug function to log data statistics and samples to wandb, then quit.
    Expects image data.
    """
    print("Debug mode: Analyzing and logging data...")
    
    # Get the dataloader
    dataset.setup('train')
    train_loader = dataset.train_dataloader()
    
    # Analyze several batches to get comprehensive statistics
    batch_stats = []
    sample_batches = []
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 5:  # Analyze first 5 batches
            break
            
        sample_batches.append(batch)
        
        # Extract input data (assuming batch[0] is input)
        if isinstance(batch, (list, tuple)):
            data = batch[0]
        else:
            raise ValueError(f"Unexpected batch format: {type(batch)}")

        # Basic statistics
        stats = {
            'batch_idx': batch_idx,
            'shape': list(data.shape),
            'dtype': str(data.dtype),
            'device': str(data.device),
            'min': float(data.min()),
            'max': float(data.max()),
            'mean': float(data.mean()),
            'std': float(data.std()),
            'nan_count': int(torch.isnan(data).sum()),
            'inf_count': int(torch.isinf(data).sum()),
        }
        
        batch_stats.append(stats)
        print(f"Batch {batch_idx} stats: {stats}")
    
    # Log general statistics
    logger.experiment.log({
        "debug/batch_statistics": batch_stats,
        "debug/dataset_info": {
            "loader_config": utils.config.to_dict(config.loader, recursive=True),
            "dataset_config": utils.config.to_dict(config.dataset, recursive=True),
            "num_batches_analyzed": len(batch_stats),
        }
    })
    
    # Handle different data types for visualization
    first_batch = sample_batches[0]
    if isinstance(first_batch, (list, tuple)):
        data = first_batch[0]
        if len(first_batch) > 1:
            targets = first_batch[1]
        else:
            raise ValueError(f"Unexpected batch format: {type(first_batch)}")
    else:
        raise ValueError(f"Unexpected batch format: {type(first_batch)}")
    
    # Determine data type from config rather than shape (since data is flattened to (B, L, D))
    dataset_name = config.dataset._name_ if hasattr(config.dataset, '_name_') else str(config.dataset)
    
    # Check if it's image data based on dataset name or config
    is_image_data = any(keyword in dataset_name.lower() for keyword in ['cifar', 'mnist', 'imagenet', 'image'])
    
    if is_image_data:
        print('logging image data (potentially language modeling)')
        _log_image_data(data, targets, logger, config)
    else:
        print('No image keywords found in dataset name, skipping.')
    
    # Log dataloader pipeline info
    _log_dataloader_info(dataset, config, logger)
    
    print("Debug logging complete. Exiting...")
    exit(0)

def _log_image_data(data, targets, logger, config):
    """Log image data samples and statistics - handles flattened image data"""
    print("Detected image data, logging image samples...")
    
    # Convert to numpy 
    print('data.shape=', data.shape)
    print('targets.shape=', targets.shape)
    if len(data.shape) == 3:
        # grayscale
        img_height, img_width = data.shape[1:]
        img_channels = 1
    else:
        a,b,c = data.shape[1:]
        # change to channel-last
        if a < b and a < c:
            data = data.permute(0, 2, 3, 1)  # (B, H, W, C)
        img_height, img_width, img_channels = data.shape[1:]

    data_np = data.cpu().numpy()  
    targets_np = targets.cpu().numpy() if targets is not None else None
    
    # Take first few samples
    num_samples = min(4, data.shape[0])  # Reduced to make room for targets
    
    fig, axes = plt.subplots(1, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    def _reshape_and_plot_image(data_np, sample_idx, axis, title, is_target=False):
        """Helper function to reshape and plot a single image sample"""
        # Extract image data, handling mask dimension if present
        img_data = data_np[sample_idx]
        
        # Reshape to image
        img_flat = img_data.flatten()
        if img_channels == 1:
            img = img_flat[:img_height*img_width].reshape(img_height, img_width)
            cmap = 'gray'
        else:
            img = img_flat[:img_height*img_width*img_channels].reshape(img_height, img_width, img_channels)
            cmap = None

        # Handle ignore values in targets (e.g., -10000)
        #if is_target:
        #    img[img == -10000.] = 0.

        # Normalize to [0, 1] if needed
        if img.max() > 1.0 or img.min() < 0.0:
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            
        print('img.shape=', img.shape)
        axis.imshow(img, cmap=cmap)
        axis.set_title(title)
        axis.axis('off')

    for i in range(num_samples):
        # Plot input image
        _reshape_and_plot_image(data_np, i, axes[i], f'Input {i}', is_target=False)
        
        # Simple classification case - update title with target value
        if targets_np.ndim == 1:
            axes[i].set_title(f'Input {i}, Target: {targets_np[i]:.3f}')
        else:
            axes[i].set_title(f'Input {i}, Target shape: {targets.shape}')
    
    plt.tight_layout()
    
    # Prepare logging data
    log_data = {
        "debug/image_samples": wandb.Image(fig),
        "debug/image_stats": {
            "original_shape": list(data.shape),
            "reconstructed_dims": [img_height, img_width, img_channels],
            "pixel_value_range": [float(data.min()), float(data.max())],
        }
    }
    
    if targets is not None:
        log_data["debug/image_stats"]["target_shape"] = list(targets.shape)
        log_data["debug/image_stats"]["target_range"] = [float(targets.min()), float(targets.max())]
        log_data["debug/image_stats"]["targets_same_shape_as_input"] = targets.shape == data.shape[:2]  # Same (B, L)
    
    # Log to wandb
    logger.experiment.log(log_data)
    plt.close(fig)

def _log_dataloader_info(dataset, config, logger):
    """Log information about the dataloader pipeline"""
    print("Logging dataloader pipeline information...")
    
    # Get dataset info
    dataset_info = {
        "dataset_class": str(type(dataset).__name__),
        "dataset_config": utils.config.to_dict(config.dataset),
        "loader_config": utils.config.to_dict(config.loader),
    }
    
    # Try to get additional dataset info
    if hasattr(dataset, '__len__'):
        try:
            dataset_info["dataset_length"] = len(dataset)
        except:
            dataset_info["dataset_length"] = "unknown"
    
    # Get dataloader info
    train_loader = dataset.train_dataloader()
    dataloader_info = {
        "batch_size": train_loader.batch_size,
        "num_workers": train_loader.num_workers,
        "pin_memory": train_loader.pin_memory,
        "drop_last": train_loader.drop_last,
        "shuffle": getattr(train_loader, 'shuffle', 'unknown'),
    }
    
    # Get transform/preprocessing info if available
    transform_info = {}
    if hasattr(dataset, 'transform') and dataset.transform is not None:
        transform_info["transforms"] = str(dataset.transform)
    if hasattr(dataset, 'target_transform') and dataset.target_transform is not None:
        transform_info["target_transforms"] = str(dataset.target_transform)
        
    # Get preprocessing pipeline info from config
    preprocessing_info = {}
    if hasattr(config.dataset, 'permute'):
        preprocessing_info["permute"] = config.dataset.permute
    if hasattr(config.dataset, 'normalization'):
        preprocessing_info["normalization"] = str(config.dataset.normalization)
    if hasattr(config.dataset, 'augment'):
        preprocessing_info["augment"] = config.dataset.augment
    
    logger.experiment.log({
        "debug/dataloader_info": {
            "dataset": dataset_info,
            "dataloader": dataloader_info,
            "transforms": transform_info,
            "preprocessing": preprocessing_info,
        }
    })
    
    # Get dataloader info
    train_loader = dataset.train_dataloader()
    loader_info = {
        "batch_size": train_loader.batch_size,
        "num_workers": train_loader.num_workers,
        "pin_memory": train_loader.pin_memory,
        "drop_last": train_loader.drop_last,
    }
    
    logger.experiment.log({
        "debug/dataset_pipeline": {
            "dataset_info": dataset_info,
            "dataloader_info": loader_info,
        }
    })