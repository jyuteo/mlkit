from dataclasses import dataclass, asdict, field
from typing import List, Optional


class Config:
    def __str__(self):
        return "\n".join(f"{key}: {value}" for key, value in self.to_dict().items())

    def to_dict(self):
        return asdict(self)


@dataclass
class LogConfig(Config):
    experiment_log_dir: str = "./logs"
    metrics_log_path: str = "./logs/metrics_log.json"


@dataclass
class WandBConfig(Config):
    enabled: bool = True
    project: Optional[str] = None
    group: Optional[str] = None
    tags: Optional[List] = None
    name: Optional[str] = None
    job_type: Optional[str] = None
    entity: Optional[str] = None
    save_code: Optional[bool] = False


@dataclass
class DataLoaderConfig(Config):
    batch_size: int = 64
    num_workers: int = 4


@dataclass
class OptimizerConfig(Config):
    weight_decay: float = 0.001


@dataclass
class LRSchedulerConfig(Config):
    step_size: int
    gamma: float = 0.01


@dataclass
class ModelCheckpointConfig(Config):
    """
    saves model state dicts throughout training phase at defined steps
    if step_by_epoch = True, and checkpoint_every = 2, then model state dict will be saved every 2 epochs
    if step_by_epoch = False, and checkpoint_every = 2, then model state dict will be saved every 2 train steps
    """  # noqa: E501

    checkpoint_every: int
    save_dir: str = "./checkpoints"


@dataclass
class ModelSnapshotConfig(Config):
    """
    saves latest model state dict at current train step, mainly for resume of interrupted training
    snapshot will be deleted when training is finished
    """

    snapshot_every: int
    save_path: str = "./snapshots/snapshot.t7"


@dataclass
class ResumeTrainingConfig(Config):
    """
    if enabled, training will resume from the model state dict in the path provided
    else, training will start from scratch
    """

    enabled: bool = False
    model_state_dict_path: str = ""


@dataclass
class TrainConfig(Config):
    # required configs
    train_epochs: int
    validate_every: int
    learning_rate: float
    lr_scheduler: LRSchedulerConfig
    model_checkpoint: ModelCheckpointConfig
    model_snapshot: ModelSnapshotConfig

    # default config values will be used if not provided
    env_vars_file_path: str = ".env"
    step_by_epoch: bool = False
    log: LogConfig = field(default_factory=LogConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    resume_training: ResumeTrainingConfig = field(default_factory=ResumeTrainingConfig)


@ dataclass
class EvaluationConfig(Config):
    pass


@dataclass
class InferenceConfig(Config):
    pass
