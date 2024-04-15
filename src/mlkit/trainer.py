import os
import wandb
import torch
import pandas as pd

from typing import List, Dict, Tuple, Any, Optional, Union
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from dotenv import load_dotenv

from .loggers.logger import Logger
from .loggers.metrics_logger import MetricsLogger
from .loggers.wandb_logger import WandBLogger
from .utils.ddp_utils import DDPUtils, NoDuplicateDistributedSampler
from .configs import (
    TrainConfig,
    EvaluationConfig,
    InferenceConfig,
    JobType,
    ModelFileType,
)


class Trainer:
    def __init__(
        self,
        config: Union[TrainConfig, EvaluationConfig],
        **kwargs,
    ):
        if isinstance(config, TrainConfig):
            self.job_type = JobType.TRAIN
        elif isinstance(config, EvaluationConfig):
            self.job_type = JobType.EVALUATION
        elif isinstance(config, InferenceConfig):
            self.job_type = JobType.INFERENCE
        else:
            raise ValueError(
                "Invalid config type. Expected TrainConfig, EvaluationConfig, or InferenceConfig."
            )
        self.config = config

        self._load_update_env_variables(self.config.env_vars_file_path)

        self.rank = DDPUtils.get_rank()
        self.is_distributed_job = self.rank is not None
        self.is_master_process = (
            not self.is_distributed_job or DDPUtils.is_master_process()
        )

        dir = Path(self.config.log.experiment_log_dir)
        dir.mkdir(parents=True, exist_ok=True)
        experiment_log_filepath = os.path.join(
            self.config.log.experiment_log_dir,
            f"log.device_{DDPUtils.get_device()}.json",
        )
        self.logger = Logger(experiment_log_filepath, self.is_master_process)

        self.metrics_logger = MetricsLogger(self.config.log.metrics_log_path)

        self.wandb_logger = None
        if self.config.wandb.enabled:
            self.wandb_logger = WandBLogger(
                is_master_process=self.is_master_process,
                logger=self.logger,
                config=self.config.to_dict(),
                **self.config.wandb.to_dict(),
            )
            if self.wandb_logger.login():
                self.wandb_logger.start()
            else:
                raise RuntimeError("Failed to login to WandB")

        self.logger.info(
            {
                "msg": "Distributed job info",
                "job_type": self.job_type.value,
                "is_distributed_job": self.is_distributed_job,
                "is_master_process": self.is_master_process,
                "rank": self.rank,
            }
        )

        self.current_train_epoch = 0
        self.current_train_step = 0

        self.train_step_results: Dict = dict()
        self.validation_step_results_for_an_epoch: List[Dict] = list()
        self.inference_step_results_for_an_epoch: List[Dict] = list()

        if self.job_type == JobType.TRAIN:
            self._init_train_variables()
        else:
            self._init_evaluation_inference_variables()

    def __del__(self):
        if self.wandb_logger:
            self.wandb_logger.close()

    def _load_update_env_variables(self, env_vars_file_path: str):
        load_dotenv(env_vars_file_path, override=True)

    def _init_train_variables(self):
        assert isinstance(
            self.config, TrainConfig
        ), "Invalid type for config. Expected TrainConfig"

        self.train_epochs = self.config.train_epochs
        self.step_by_epoch = self.config.step_by_epoch
        self.checkpoint_every = self.config.model_checkpoint.checkpoint_every
        self.validate_every = self.config.validate_every
        self.snapshot_every = self.config.model_snapshot.snapshot_every

        self.train_dataset, self.val_dataset = self.build_dataset()
        self.train_dataset_sampler, self.val_dataset_sampler = (
            self.build_dataset_sampler()
        )
        self.batch_size = self.config.dataloader.batch_size
        self.num_workers = self.config.dataloader.num_workers
        self.train_dataloader, self.val_dataloader = self.build_dataloader()

        self.model = DDPUtils.move_model_to_device(self.build_model())

        self.learning_rate = self.config.learning_rate
        self.optimizer = self.build_optimizer(**self.config.optimizer.to_dict())
        self.lr_scheduler = self.build_lr_scheduler(
            **self.config.lr_scheduler.to_dict()
        )

        self.model_checkpoint_dir = self.config.model_checkpoint.save_dir
        dir = Path(self.model_checkpoint_dir)
        dir.mkdir(parents=True, exist_ok=True)

        self.model_snapshot_dir = self.config.model_snapshot.save_dir
        dir = Path(self.model_snapshot_dir)
        dir.mkdir(parents=True, exist_ok=True)
        self.model_snapshot_path = os.path.join(self.model_snapshot_dir, "snapshot.t7")
        if os.path.exists(self.model_snapshot_path):
            self._load_state_dict_to_resume_training(self.model_snapshot_path)
            self.logger.info(
                {
                    "msg": f"Resuming failed training from snapshot: {self.model_snapshot_path}",  # noqa: E501
                    "train_epoch": self.current_train_epoch,
                    "train_step": self.current_train_step,
                }
            )
        elif self.config.resume_training.enabled:
            self._load_state_dict_to_resume_training(
                self.config.resume_training.model_state_dict_path
            )
            self.logger.info(
                {
                    "msg": f"Resuming training from given model state dict: {self.config.resume_training.model_state_dict_path}",  # noqa: E501
                    "train_epoch": self.current_train_epoch,
                    "train_step": self.current_train_step,
                }
            )

    def _init_evaluation_inference_variables(self):
        assert (
            self.config.model_file.file_path
            and os.path.exists(self.config.model_file.file_path)
            and self.config.model_file.file_type
            in (ModelFileType.STATE_DICT, ModelFileType.TORCHSCRIPT)
        ), "Invalid model state dicts path"

        self.model_file_path = self.config.model_file.file_path
        self.train_dataset, self.val_dataset = self.build_dataset()
        self.train_dataset_sampler, self.val_dataset_sampler = (
            self.build_dataset_sampler()
        )
        self.batch_size = self.config.dataloader.batch_size
        self.num_workers = self.config.dataloader.num_workers
        self.train_dataloader, self.val_dataloader = self.build_dataloader()

        if self.config.model_file.file_type == ModelFileType.STATE_DICT:
            self.model = DDPUtils.move_model_to_device(self.build_model())
            self._load_state_dict_for_evaluation_inference(self.model_file_path)
        else:
            scripted_model = self.load_model_torchscript(self.model_file_path)
            self.model = DDPUtils.move_model_to_device(scripted_model)
        self.logger.info(f"Loaded model file from {self.model_file_path}")

    def _load_state_dict_to_resume_training(self, state_dict_path: str):
        assert os.path.exists(
            state_dict_path
        ), f"State dicts path does not exist: {state_dict_path}"

        map_location = None
        if self.is_distributed_job:
            map_location = {"cuda:0": f"cuda:{self.rank}"}
        state_dict = torch.load(state_dict_path, map_location=map_location)

        required_keys = ["model", "optimizer", "lr_scheduler", "epoch", "step"]
        assert all(
            [key in state_dict for key in required_keys]
        ), f"Invalid state dicts. Required keys: {required_keys}"

        if self.is_distributed_job:
            self.model.module.load_state_dict(state_dict["model"])
        else:
            self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
        self.current_train_epoch = state_dict["epoch"]
        self.current_train_step = state_dict["step"]

    def _load_state_dict_for_evaluation_inference(self, model_file_path: str):
        assert os.path.exists(
            model_file_path
        ), f"State dicts path does not exist: {model_file_path}"

        map_location = None
        if self.is_distributed_job:
            map_location = {"cuda:0": f"cuda:{self.rank}"}
        state_dict = torch.load(model_file_path, map_location=map_location)

        required_keys = ["model"]
        assert all(
            [key in state_dict for key in required_keys]
        ), f"Invalid state dicts. Required keys: {required_keys}"

        if self.is_distributed_job:
            self.model.module.load_state_dict(state_dict["model"])
        else:
            self.model.load_state_dict(state_dict["model"])

    def load_model_torchscript(self, model_script_path: str) -> torch.nn.Module:
        assert os.path.exists(
            model_script_path
        ), f"Model script path does not exist: {model_script_path}"

        return torch.jit.load(model_script_path)

    def _delete_model_snapshot(self):
        if os.path.exists(self.model_snapshot_path):
            os.remove(self.model_snapshot_path)

    def _set_ddp_barrier(self) -> None:
        if self.is_distributed_job:
            DDPUtils.set_barrier()

    def build_dataset(self) -> Tuple[Optional[Dataset], Optional[Dataset]]:
        """
        Method to construct train, evaluation or inference dataset

        Returns:
            Tuple[Optional[Dataset], Optional[Dataset]]: The first element is the dataset used for training,
            while the second element is the dataset used for either evaluation or inference.
            If job type is EVALUATION or INFERENCE, first element in the tuple should be None
        """  # noqa: E501
        raise NotImplementedError

    def build_dataset_sampler(
        self,
    ) -> Tuple[
        Optional[NoDuplicateDistributedSampler], Optional[NoDuplicateDistributedSampler]
    ]:
        train_dataset_sampler, val_dataset_sampler = None, None
        if self.is_distributed_job:
            if self.train_dataset:
                train_dataset_sampler = NoDuplicateDistributedSampler(
                    self.train_dataset
                )
            if self.val_dataset:
                val_dataset_sampler = NoDuplicateDistributedSampler(
                    self.val_dataset, shuffle=False
                )
        return train_dataset_sampler, val_dataset_sampler

    def build_dataloader(self) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        """
        Method to construct train, evaluation or inference dataloader

        Returns:
            Tuple[Optional[DataLoader], Optional[DataLoader]]: The first element is the dataloader used for training,
            while the second element is the dataloader used for either evaluation or inference.
            If job type is EVALUATION or INFERENCE, first element in the tuple should be None
        """  # noqa: E501
        train_dataloader, val_dataloader = None, None
        if self.train_dataset:
            train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                shuffle=self.train_dataset_sampler is None,
                sampler=self.train_dataset_sampler,
            )
        if self.val_dataset:
            val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                shuffle=False,
                sampler=self.val_dataset_sampler,
            )
        return train_dataloader, val_dataloader

    def build_model(self) -> torch.nn.Module:
        if (
            self.job_type in (JobType.EVALUATION, JobType.INFERENCE)
            and self.config.model_file.file_type == ModelFileType.TORCHSCRIPT
        ):
            return
        raise NotImplementedError

    def build_optimizer(self, **kwargs) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
        )

    def build_lr_scheduler(
        self,
        step_size: int,
        gamma: float = 0.1,
        **kwargs,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size, gamma)

    def train_step(self, batch_data: Any) -> Dict:
        if self.job_type == JobType.TRAIN:
            raise NotImplementedError

    def _do_on_train_step_start(self):
        self.current_train_step += 1
        self.train_step_results = dict()
        self._set_ddp_barrier()
        if not self.step_by_epoch:
            self.logger.debug(
                {
                    "msg": "Train step started",
                    "train_epoch": self.current_train_epoch,
                    "train_step": self.current_train_step,
                }
            )
        self.do_on_train_step_start()

    def do_on_train_step_start(self) -> None:
        pass

    def _do_on_train_step_end(self) -> None:
        self._set_ddp_barrier()
        train_step_losses = DDPUtils.all_gather_tensors(self.train_step_results["loss"])
        if self.is_master_process:
            train_step_metrics = {
                "loss": torch.mean(train_step_losses),
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            self.metrics_logger.log(
                train_step_metrics, self.current_train_step, "train"
            )
            self.log_wandb_metrics(train_step_metrics, "train")
        if not self.step_by_epoch and self.is_master_process:
            if self.current_train_step % self.checkpoint_every == 0:
                self.checkpoint_model()
            if self.current_train_step % self.snapshot_every == 0:
                self._save_model_snapshot()
        if self.current_train_step % self.validate_every == 0:
            self._run_validation_epoch()
        self.do_on_train_step_end()

    def do_on_train_step_end(self) -> None:
        pass

    def _run_train_epoch(self) -> None:
        self._do_on_train_epoch_start()

        for batch in self.train_dataloader:
            self._do_on_train_step_start()

            self.model.train()
            self.optimizer.zero_grad()

            batch = DDPUtils.move_tensor_or_tuple_of_tensors_to_device(batch)
            train_step_results = self.train_step(batch)
            assert (
                "loss" in train_step_results
            ), "Key 'loss' not found in train step results"
            assert isinstance(
                train_step_results["loss"], torch.Tensor
            ), "Value corresponding to key 'loss' is not a PyTorch tensor"

            self.train_step_results = train_step_results
            train_loss = train_step_results["loss"]
            train_loss.backward()

            self.optimizer.step()

            if not self.step_by_epoch:
                self.lr_scheduler.step()

            self._do_on_train_step_end()

        if self.step_by_epoch:
            self.lr_scheduler.step()

        self._do_on_train_epoch_end()

    def _do_on_train_epoch_start(self) -> None:
        self.current_train_epoch += 1
        self.logger.info(
            {
                "msg": "Train epoch started",
                "train_epoch": self.current_train_epoch,
                "train_step": self.current_train_step,
            }
        )
        if self.is_distributed_job:
            self.train_dataset_sampler.set_epoch(self.current_train_epoch)
        self.do_on_train_epoch_start()

    def do_on_train_epoch_start(self) -> None:
        pass

    def _do_on_train_epoch_end(self) -> None:
        if self.step_by_epoch and self.is_master_process:
            if self.current_train_epoch % self.checkpoint_every == 0:
                self.checkpoint_model()
            if self.current_train_epoch % self.snapshot_every == 0:
                self._save_model_snapshot()
        if self.current_train_epoch % self.validate_every == 0:
            self._run_validation_epoch()
        self.do_on_train_epoch_end()

    def do_on_train_epoch_end(self) -> None:
        pass

    def train(self) -> None:
        assert (
            self.job_type == JobType.TRAIN
        ), f"Invalid job config type. Expected {TrainConfig.__name__}"

        for _ in range(self.current_train_epoch + 1, self.train_epochs + 1):
            self._run_train_epoch()
        if self.is_master_process:
            self._delete_model_snapshot()

    def validation_step(self, batch_data: Any) -> Optional[Dict]:
        """
        Return value will be appened to self.validation_step_results_for_an_epoch,
        and can be used to calculate metrics for the a epoch in later step
        """
        if self.job_type in (JobType.TRAIN, JobType.EVALUATION):
            raise NotImplementedError

    def _do_on_validation_step_start(self) -> None:
        self._set_ddp_barrier()
        self.do_on_validation_step_start()

    def do_on_validation_step_start(self) -> None:
        pass

    def _do_on_validation_step_end(self) -> None:
        self._set_ddp_barrier()
        self.do_on_validation_step_end()

    def do_on_validation_step_end(self) -> None:
        pass

    def _run_validation_epoch(self) -> None:
        self._do_on_validation_epoch_start()
        self.model.eval()

        with torch.no_grad():
            for batch in self.val_dataloader:
                self._do_on_validation_step_start()
                batch = DDPUtils.move_tensor_or_tuple_of_tensors_to_device(batch)
                result = self.validation_step(batch)
                if result:
                    self.validation_step_results_for_an_epoch.append(result)
                self._do_on_validation_step_end()

        self._do_on_validation_epoch_end()

    def _do_on_validation_epoch_start(self) -> None:
        self.validation_step_results_for_an_epoch = list()
        self.logger.debug(
            {
                "msg": "Validation started",
                "train_epoch": self.current_train_epoch,
                "train_step": self.current_train_step,
            }
        )
        self.do_on_validation_epoch_start()

    def do_on_validation_epoch_start(self) -> None:
        pass

    def _do_on_validation_epoch_end(self) -> None:
        self.do_on_validation_epoch_end()

    def do_on_validation_epoch_end(self) -> None:
        pass

    def evaluate(self) -> None:
        assert (
            self.job_type == JobType.EVALUATION
        ), f"Invalid job config type. Expected {EvaluationConfig.__name__}"

        self._run_validation_epoch()

    def inference_step(self, batch_data: Any) -> Optional[Dict]:
        """
        Return value will be appened to self.inference_step_results_for_an_epoch,
        and can be used to calculate metrics for the a epoch in later step
        """
        if self.job_type == JobType.INFERENCE:
            raise NotImplementedError

    def _do_on_inference_step_start(self) -> None:
        self._set_ddp_barrier()
        self.do_on_inference_step_start()

    def do_on_inference_step_start(self) -> None:
        pass

    def _do_on_inference_step_end(self) -> None:
        self._set_ddp_barrier()
        self.do_on_inference_step_end()

    def do_on_inference_step_end(self) -> None:
        pass

    def _run_inference_epoch(self) -> None:
        self._do_on_inference_epoch_start()
        self.model.eval()

        with torch.no_grad():
            for batch in self.val_dataloader:
                self._do_on_inference_step_start()
                batch = DDPUtils.move_tensor_or_tuple_of_tensors_to_device(batch)
                result = self.inference_step(batch)
                if result:
                    self.inference_step_results_for_an_epoch.append(result)
                self._do_on_inference_step_end()

        self._do_on_inference_epoch_end()

    def _do_on_inference_epoch_start(self) -> None:
        self.inference_step_results_for_an_epoch = list()
        self.logger.info("Inference started")
        self.do_on_inference_epoch_start()

    def do_on_inference_epoch_start(self) -> None:
        pass

    def _do_on_inference_epoch_end(self) -> None:
        self.do_on_inference_epoch_end()

    def do_on_inference_epoch_end(self) -> None:
        pass

    def inference(self) -> None:
        assert (
            self.job_type == JobType.INFERENCE
        ), f"Invalid job config type. Expected {InferenceConfig.__name__}"

        self._run_inference_epoch()

    def save_model_state_dict(
        self,
        save_path: str,
        model: torch.nn.Module = None,
        log_model_to_wandb: bool = True,
        wandb_model_name: Optional[str] = None,
    ) -> None:
        assert self.is_master_process, "Only save model state dicts in master process"
        if model is not None:
            model_to_save = model
        else:
            model_to_save = (
                self.model.module.state_dict()
                if self.is_distributed_job
                else self.model.state_dict()
            )
        state = {
            "epoch": self.current_train_epoch,
            "step": self.current_train_step,
            "model": model_to_save,
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }
        torch.save(state, save_path)
        if self.wandb_logger and log_model_to_wandb:
            self.wandb_logger.log_model(model_path=save_path, name=wandb_model_name)

    def save_model_torchscript(
        self,
        save_path: str,
        model: torch.nn.Module = None,
        log_model_to_wandb: bool = True,
        wandb_model_name: Optional[str] = None,
    ):
        assert self.is_master_process, "Only save model state dicts in master process"
        if model is not None:
            model_to_save = model
        else:
            model_to_save = self.model.module if self.is_distributed_job else self.model
        scripted_model = torch.jit.script(model_to_save)
        scripted_model.save(save_path)
        if self.wandb_logger and log_model_to_wandb:
            self.wandb_logger.log_model(model_path=save_path, name=wandb_model_name)

    def checkpoint_model(self) -> None:
        if not self.is_master_process:
            return
        self.save_model_state_dict(
            save_path=os.path.join(
                self.model_checkpoint_dir, f"checkpoint_{self.current_train_step}.t7"
            ),
            wandb_model_name="checkpoint-state_dict",
        )
        self.save_model_torchscript(
            save_path=os.path.join(
                self.model_checkpoint_dir, f"checkpoint_{self.current_train_step}.pt"
            ),
            wandb_model_name="checkpoint-torchscript",
        )

    def save_best_model(self) -> None:
        if not self.is_master_process:
            return
        self.save_model_state_dict(
            save_path=os.path.join(self.model_checkpoint_dir, "best.t7"),
            wandb_model_name="best-state-dict",
        )
        self.save_model_torchscript(
            save_path=os.path.join(self.model_checkpoint_dir, "best.pt"),
            wandb_model_name="best-torchscript",
        )

    def _save_model_snapshot(self) -> None:
        if not self.is_master_process:
            return
        self.save_model_state_dict(
            save_path=self.model_snapshot_path,
            log_model_to_wandb=False,
        )

    def log_wandb_metrics(self, metrics: Dict, category: str = "") -> None:
        if not self.wandb_logger:
            return
        self.wandb_logger.log_metrics(metrics, self.current_train_step, category)

    def log_wandb_table(
        self, table: Union[wandb.Table, pd.DataFrame], table_name: str
    ) -> None:
        if not self.wandb_logger:
            return
        self.wandb_logger.log_table(table, table_name)

    def log_wandb_model(self, model_path: str, name: Optional[str] = None):
        if not self.wandb_logger:
            return
        self.wandb_logger.log_model(model_path, name)
