# Fast Iterated Sums

The codebase for the paper "Tensor-to-Tensor Models with Fast Iterated Sum Features" by
Joscha Diehl, Rasheed Ibraheem, Leonard Schmitz, Yue Wu (https://www.sciencedirect.com/science/article/pii/S092523122600281X).


Uses PyTorch Lightning + Hydra for training and hyperparameter configuration.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train with default config (CIFAR-10 + FCN)
python train.py

# Train with specific experiment
python train.py experiment=fis_convnext-cifar10-adamw

# Override config values
python train.py optimizer.lr=0.01 trainer.max_epochs=100

# Disable wandb logging
python train.py '~wandb'

# To just print the config and exit:
python train.py ... whatever params ...  --cfg job

# Running a sweep
chmod +x run_sweep.sh
./run_sweep.sh sweep-configs/fis_convnext-cifar10-adamw-fis-max.yaml

# Running experiment ConvNeXt tiny, L4, CIFAR10 (see Table 2 in the paper):
python train.py experiment=fis_convnext-cifar10-adamw loader.batch_size=32 model.discount_init=-0.1 model.match_dim_cnn=False model.model_mode=l4 "model.seed=[450, 475, 500]" model.semiring=maxplus model.sums_internal_activation=None model.tree_type=parameter_sharing_v2 model.use_discounted_sums=True scheduler.epochs=100 train.seed=2222 trainer.max_epochs=100 '~wandb'
```

## Contributed layers


### FISLayerFixedTree

We recommend using FISLayerFixedTree for most applications,
since it is much faster than FISLayerRandomTrees and deterministic.

For usage see tests/fis_layer_fixed_tree_test.py

### FISLayerRandomTrees

For usage see tests/fis_layer_random_trees_test.py


## Configuration

The hydra configuration structure is based on https://github.com/IdoAmos/not-from-scratch
(which itself is based on https://github.com/HazyResearch/state-spaces)
and https://github.com/MIC-DKFZ/image_classification
Base config in `configs/config.yaml`.

**Config structure:**
- `experiment/` - Complete experiment setups
- `model/` - Model architectures  
- `pipeline/` - Dataset + task configurations
- `optimizer/` - Optimizer settings
- `scheduler/` - Learning rate schedulers
- `callbacks/` - Training callbacks
- `trainer/` - PyTorch Lightning trainer config

## Outputs

Results saved to `outputs/YYYY-MM-DD/HH-MM-SS/`:
- Logs, checkpoints, metrics
- Working directory changes to output folder during run
- WandB logging enabled by default

## Instructions for Windows users
If you are using windows, `triton` is not officially available to download from PyPI.
We used the `triton-windows` instead. This can be installed (the command we used) via the following command:
```
pip install -U "triton-windows<3.6"
```

## Please cite

If you use this code in your work, please cite:

```bibtex
@article{diehl2026tensor,
  title   = {Tensor-to-tensor models with fast iterated sum features},
  author  = {Diehl, Joscha and Ibraheem, Rasheed and Schmitz, Leonard and Wu, Yue},
  journal = {Neurocomputing},
  volume  = {675},
  pages   = {132884},
  year    = {2026},
  doi     = {10.1016/j.neucom.2026.132884},
  url     = {https://www.sciencedirect.com/science/article/pii/S092523122600281X}
}
```