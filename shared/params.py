from shared.defintions import PoolMode, Semiring

SWEEP_CONFIG = {
    "method": "grid",
    "metric": {"name": "best_validation_accuracy", "goal": "maximize"},
    "parameters": {
        "batch_size": {"value": 256},
        "num_epochs": {"value": 200},
        "pool_mode": {"values": [m for m in PoolMode]},
        "num_nodes": {"value": 3},
        "seed": {"value": 42},
        "semiring": {"values": [sm for sm in Semiring]},
    },
}


CLF_TRAIN_HYPERPARAMS = {
    "batch_size": 256,
    "pool_mode": PoolMode.MAX,
    "num_nodes": 3,
    "seed": 42,
}


CLF_RANDOM_HYPERPARAMS = {
    "batch_size": 256,
    "pool_mode": PoolMode.MAX,
    "num_nodes": 3,
    "seed": 42,
    "num_repeat": 10,
}


AD_TRAIN_HYPERPARAMS = {
    "batch_size": 16,
    "latent_dim": 32,
    "num_nodes": 3,
    "seed": 42,
}


AD_BACKBONE_HYPERPARAMS = {
    "batch_size": 16,
    "latent_dim": 32,
    "num_nodes": 3,
    "seed": 42,
}


AD_LATENT_DIM_HYPERPARAMS = {
    "batch_size": 16,
    "num_nodes": 3,
    "seed": 42,
}

AD_TOPN_HYPERPARAMS = {
    "batch_size": 16,
    "num_nodes": 3,
    "latent_dim": 32,
    "seed": 42,
}
