from dataclasses import dataclass, asdict, field
from typing import List, Optional
from enum import Enum


class JobType(Enum):
    TRAIN = "train"
    EVALUATION = "evaluation"
    INFERENCE = "inference"


class ModelFileType(Enum):
    STATE_DICT = "state_dict"
    TORCHSCRIPT = "torchscript"


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
    Saves model state dicts throughout training phase at defined steps.

    If step_by_epoch = True, and checkpoint_every = 2, then model state dict will be saved every 2 epochs.
    If step_by_epoch = False, and checkpoint_every = 2, then model state dict will be saved every 2 train steps.
    """  # noqa: E501

    checkpoint_every: int
    save_dir: str = "./checkpoints"


@dataclass
class ModelSnapshotConfig(Config):
    """
    Saves latest model state dict at current train step, mainly for resume of interrupted training.
    Snapshot will be deleted when training is finished.

    If step_by_epoch = True, and snapshot_every = 2, then model snapshot will be saved every 2 epochs.
    If step_by_epoch = False, and snapshot_every = 2, then model snapshot will be saved every 2 train steps.
    """  # noqa: E501

    snapshot_every: int
    save_dir: str = "./snapshots"


@dataclass
class ResumeTrainingConfig(Config):
    """
    If enabled, training will resume from the model state dict in the path provided.
    Else, training will start from scratch.
    """

    enabled: bool = False
    model_state_dict_path: str = ""


@dataclass
class ModelFileConfig(Config):
    """
    Specifies the type of saved model to be loaded during evaluation or inference.
    It can be in state dict (.t7) or torchscript (.pt) file.
    """

    file_type: ModelFileType
    file_path: str


@dataclass
class TrainConfig(Config):
    train_epochs: int
    validate_every: int
    learning_rate: float
    lr_scheduler: LRSchedulerConfig
    model_checkpoint: ModelCheckpointConfig
    model_snapshot: ModelSnapshotConfig

    env_vars_file_path: str = ".env"
    step_by_epoch: bool = False
    log: LogConfig = field(default_factory=LogConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    resume_training: ResumeTrainingConfig = field(default_factory=ResumeTrainingConfig)


@dataclass
class EvaluationConfig(Config):
    env_vars_file_path: str = ".env"
    log: LogConfig = field(default_factory=LogConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    model_file: ModelFileConfig = field(default_factory=ModelFileConfig)


@dataclass
class InferenceConfig(Config):
    env_vars_file_path: str = ".env"
    log: LogConfig = field(default_factory=LogConfig)
    wandb: WandBConfig = WandBConfig(enabled=False)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    model_file: ModelFileConfig = field(default_factory=ModelFileConfig)
