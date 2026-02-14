optimizer = {
    "adam": "torch.optim.Adam",
    "adamw": "torch.optim.AdamW",
    "rmsprop": "torch.optim.RMSprop",
    "sgd": "torch.optim.SGD",
    "lamb": "src.utils.optim.lamb.JITLamb",
}

scheduler = {
    "onecycle": "torch.optim.lr_scheduler.OneCycleLR",
    "constant": "transformers.get_constant_schedule",
    "plateau": "torch.optim.lr_scheduler.ReduceLROnPlateau",
    "step": "torch.optim.lr_scheduler.StepLR",
    "multistep": "torch.optim.lr_scheduler.MultiStepLR",
    "cosine": "torch.optim.lr_scheduler.CosineAnnealingLR",  # T_max, eta_min=0, last_epoch=-1
    "constant_warmup": "transformers.get_constant_schedule_with_warmup",  # num_warmup_steps, last_epoch=-1
    "linear_warmup": "transformers.get_linear_schedule_with_warmup",  # num_warmup_steps, num_training_steps, last_epoch=-1
    "cosine_warmup": "transformers.get_cosine_schedule_with_warmup",  # num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1
    "timm_cosine": "src.utils.optim.schedulers.TimmCosineLRScheduler",
    "cosine_warmup_restarts": "transformers.get_cosine_with_hard_restarts_schedule_with_warmup",  # num_warmup_steps, num_training_steps, num_cycles=1, last_epoch=-1
    # ( optimizer: Optimizernum_warmup_steps: intnum_training_steps: intnum_cycles: int = 1last_epoch: int = -1 )
    # "cosine_warm_restarts": "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts",
}

model = {
    # Backbones from this repo
    "model": "src.models.sequence.SequenceModel",
    "unet": "src.models.sequence.SequenceUNet",
    "sashimi": "src.models.sequence.sashimi.Sashimi",
    "sashimi_standalone": "sashimi.sashimi.Sashimi",
    # Baseline RNNs
    "lstm": "src.models.baselines.lstm.TorchLSTM",
    "gru": "src.models.baselines.gru.TorchGRU",
    "unicornn": "src.models.baselines.unicornn.UnICORNN",
    "odelstm": "src.models.baselines.odelstm.ODELSTM",
    "lipschitzrnn": "src.models.baselines.lipschitzrnn.RnnModels",
    "stackedrnn": "src.models.baselines.samplernn.StackedRNN",
    "stackedrnn_baseline": "src.models.baselines.samplernn.StackedRNNBaseline",
    "samplernn": "src.models.baselines.samplernn.SampleRNN",
    "statsmlp": "src.models.baselines.statsmlp.StatsMLP",
    # Baseline CNNs
    "ckconv": "src.models.baselines.ckconv.ClassificationCKCNN",
    "wavegan": "src.models.baselines.wavegan.WaveGANDiscriminator",  # DEPRECATED
    "wavenet": "src.models.baselines.wavenet.WaveNetModel",
    "torch/resnet2d": "src.models.baselines.resnet.TorchVisionResnet",
    # Nonaka 1D CNN baselines
    "nonaka/resnet18": "src.models.baselines.nonaka.resnet.resnet1d18",
    "nonaka/inception": "src.models.baselines.nonaka.inception.inception1d",
    "nonaka/xresnet50": "src.models.baselines.nonaka.xresnet.xresnet1d50",
    # Pretrained models from huggingface,
    "pythia_lm": "src.models.lm.model.PythiaWrapper",
}

layer = {
    "id": "src.models.sequence.base.SequenceIdentity",
    "lstm": "src.models.sequence.rnns.lstm.TorchLSTM",
    "sru": "src.models.sequence.rnns.sru.SRURNN",
    "lssl": "src.models.sequence.ss.lssl.LSSL",
    "s4": "src.models.sequence.ss.s4.S4",
    "standalone": "src.models.s4.s4.S4",
    "s4d": "src.models.s4.s4d.S4D",
    "dlr": "src.models.sequence.ss.dss.DSS",
    "ff": "src.models.sequence.ff.FF",
    "rnn": "src.models.sequence.rnns.rnn.RNN",
    "mha": "src.models.sequence.mha.MultiheadAttention",
    "mhfa": "src.models.sequence.mha.MultiheadAttentionFlash",
    "mhfla": "src.models.sequence.mha.MultiheadLocalAttention",
    "conv1d": "src.models.sequence.convs.conv1d.Conv1d",
    "conv2d": "src.models.sequence.convs.conv2d.Conv2d",
    "performer": "src.models.sequence.attention.linear.Performer",
    "linear_attention_2": "src.models.sequence.attention.linear_attention_2.LinearAttention",
    "linear_attention_2_norm": "src.models.sequence.attention.linear_attention_2.LinearAttentionWithNorm",
    "linear_attention_lra": "src.models.sequence.attention.linear_attention_lra.LinearAttention",
    "mega": "src.models.sequence.mega.MegaBlock",
    "mamba2_start_layer": "src.models.mamba.mamba2_start.Mamba2StartLayer",
    "mamba2_from_ssm_benchmark_layer": "src.models.mamba.mamba2_from_ssm_benchmark.Mamba2FromSSMBenchmarkLayer",
    "mamba2_ca_layer": "src.models.mamba.mamba2_cosattention.Mamba2CosAttentionLayer",
    "mamba2_ca_layer_stacked": "src.models.mamba.mamba2_cosattention_stacked.Mamba2CosAttentionLayer",
    "cos_attention": "src.models.sequence.attention.cos_attention.CosAttention",
}

callbacks = {
    "timer": "src.callbacks.timer.Timer",
    "params": "src.callbacks.params.ParamsLog",
    "learning_rate_monitor": "lightning.pytorch.callbacks.LearningRateMonitor",
    "model_checkpoint": "lightning.pytorch.callbacks.ModelCheckpoint",
    "early_stopping": "lightning.pytorch.callbacks.EarlyStopping",
    "swa": "lightning.pytorch.callbacks.StochasticWeightAveraging",
    "rich_model_summary": "lightning.pytorch.callbacks.RichModelSummary",
    "rich_progress_bar": "lightning.pytorch.callbacks.RichProgressBar",
    "progressive_resizing": "src.callbacks.progressive_resizing.ProgressiveResizing",
    "log_run_info": "src.callbacks.log_run_info.LogRunInfo",
    "running_max_logger": "src.callbacks.running_max_logger.RunningMaxLogger",
    "log_image_predictions": "src.callbacks.wandb.LogImagePredictions",
    "log_fis_images": "src.callbacks.wandb.LogFISImages",
    "log_cnn_images": "src.callbacks.wandb.LogCNNImages",
    "log_gamma_parameters": "src.callbacks.wandb.LogGammaParameters",
    "log_lambda_parameters": "src.callbacks.wandb.LogLambdaParameters",
    "log_positional_encoding": "src.callbacks.wandb.LogPositionalEncoding",
}

layer_decay = {
    'convnext_tiny': 'src.models.fis_resnet_v1.get_num_layer_for_convnext_tiny',
    'fcn': 'src.models.fcn.get_num_layer_for_fcn',
}