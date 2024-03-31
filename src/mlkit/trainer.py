import os
import torch

from typing import List, Dict, Tuple, Any
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP

from .logger import Logger
from .metrics_logger import MetricsLogger
from .utils.ddp_utils import DDPUtils


class Trainer:
    def __init__(
        self,
        train_epochs: int,
        dataloader_batch_size: int,
        dataloader_num_workers: int,
        learning_rate: float,
        checkpoint_every: int,
        optimizer_config: Dict,
        lr_scheduler_config: Dict,
        step_by_epoch: bool = True,
        validate_every: int = 1,
        resume_training: bool = False,
        resume_training_model_state_dict_path: str = "",
        experiment_log_dir: str = "./logs",
        metrics_log_filepath: str = "./logs/metrics_log.json",
        model_checkpoint_dir: str = "./checkpoints",
        model_snapshot_path: str = "./snapshot.t7",
        **kwargs,
    ):
        config = locals()
        config.pop("self")

        dir = Path(experiment_log_dir)
        dir.mkdir(parents=True, exist_ok=True)
        experiment_log_filepath = os.path.join(
            experiment_log_dir, f"train_log.device_{DDPUtils.get_device()}.json"
        )
        self.logger = Logger(experiment_log_filepath)
        self.logger.log({"msg": "Config", **config})

        self.rank = DDPUtils.get_rank()
        self.is_distributed_training = self.rank is not None
        self.is_master_process = (
            not self.is_distributed_training or DDPUtils.is_master_process()
        )
        self.logger.log(
            {
                "msg": "Distributed training",
                "is_distributed_training": self.is_distributed_training,
                "is_master_process": self.is_master_process,
                "rank": self.rank,
            }
        )

        self.train_epochs = train_epochs
        self.step_by_epoch = step_by_epoch
        self.checkpoint_every = checkpoint_every
        self.validate_every = validate_every

        self.batch_size = dataloader_batch_size
        self.num_workers = dataloader_num_workers
        self.train_dataset_sampler, self.val_dataset_sampler = (
            self.build_dataset_sampler()
        )
        self.train_dataset, self.val_dataset = self.build_dataset()
        self.train_dataloader, self.val_dataloader = self.build_dataloader()

        self.model = DDPUtils.move_model_to_device(self.build_model())

        self.learning_rate = learning_rate
        self.optimizer = self.build_optimizer(**optimizer_config)
        self.lr_scheduler = self.build_lr_scheduler(**lr_scheduler_config)

        self.current_train_epoch = 0
        self.current_train_step = 0

        self.train_step_results: Dict = dict()
        self.validation_step_results_for_an_epoch: List[Dict] = list()

        self.metrics_logger = (
            MetricsLogger(metrics_log_filepath) if self.is_master_process else None
        )

        self.model_checkpoint_dir = model_checkpoint_dir
        dir = Path(self.model_checkpoint_dir)
        dir.mkdir(parents=True, exist_ok=True)

        self.model_snapshot_path = model_snapshot_path
        if os.path.exists(self.model_snapshot_path):
            self._load_state_dicts_to_resume_training(self.model_snapshot_path)
            self.logger.log(
                {
                    "msg": f"Resuming failed training from snapshot: {self.model_snapshot_path}",
                    "train_epoch": self.current_train_epoch,
                    "train_step": self.current_train_step,
                }
            )
        elif resume_training:
            self._load_state_dicts_to_resume_training(
                resume_training_model_state_dict_path
            )
            self.logger.log(
                {
                    "msg": f"Resuming training from given model state dict: {resume_training_model_state_dict_path}",  # noqa: E501
                    "train_epoch": self.current_train_epoch,
                    "train_step": self.current_train_step,
                }
            )

    def _load_state_dicts_to_resume_training(self, checkpoint_path: str):
        assert os.path.exists(
            checkpoint_path
        ), f"Checkpoint path does not exist: {checkpoint_path}"

        map_location = None
        if self.is_distributed_training:
            map_location = {"cuda:0": f"cuda:{self.rank}"}
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        self.current_train_epoch = checkpoint["epoch"]
        self.current_train_step = checkpoint["step"]

    def _delete_model_snapshot(self):
        if os.path.exists(self.model_snapshot_path):
            os.remove(self.model_snapshot_path)

    def _set_ddp_barrier(self) -> None:
        if self.is_distributed_training:
            DDP.set_barrier()

    def build_dataset(self) -> Tuple[Dataset, Dataset]:
        raise NotImplementedError

    def build_dataset_sampler(self):
        train_dataset_sampler, val_dataset_sampler = None, None
        if self.is_distributed_training:
            train_dataset_sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_dataset
            )
            val_dataset_sampler = torch.utils.data.distributed.DistributedSampler(
                self.val_dataset, shuffle=False, drop_last=True
            )
        return train_dataset_sampler, val_dataset_sampler

    def build_dataloader(self) -> Tuple[DataLoader, DataLoader]:
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=self.train_dataset_sampler is None,
            sampler=self.train_dataset_sampler,
        )
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

    def do_on_train_step_start(self):
        pass

    def _do_on_train_step_end(self):
        self._set_ddp_barrier()
        if not self.step_by_epoch and self.is_master_process:
            if self.current_train_step % self.checkpoint_every == 0:
                self.save_model_state_dicts()
            if self.current_train_step % self.validate_every == 0:
                self._run_validation_epoch()
            self.save_model_snapshot()
        self.do_on_train_step_end()

    def do_on_train_step_end(self):
        pass

    def validation_step(self, batch_data: Any) -> Dict:
        raise NotImplementedError

    def _do_on_validation_step_start(self):
        self.do_on_validation_step_start()

    def do_on_validation_step_start(self):
        pass

    def _do_on_validation_step_end(self):
        self.do_on_validation_step_end()

    def do_on_validation_step_end(self):
        pass

    def _run_train_epoch(self):
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

    def _do_on_train_epoch_start(self):
        self.current_train_epoch += 1
        self.logger.log(
            {
                "msg": "Train epoch started",
                "train_epoch": self.current_train_epoch,
                "train_step": self.current_train_step,
            }
        )
        if self.is_distributed_training:
            self.train_dataset.sampler.set_epoch(self.current_train_epoch)
        self.do_on_train_epoch_start()

    def do_on_train_epoch_start(self):
        pass

    def _do_on_train_epoch_end(self):
        if self.step_by_epoch and self.is_master_process:
            if self.current_train_epoch % self.checkpoint_every == 0:
                self.save_model_state_dicts()
            if self.current_train_epoch % self.validate_every == 0:
                self._run_validation_epoch()
            self.save_model_snapshot()
        self.do_on_train_epoch_end()

    def do_on_train_epoch_end(self):
        pass

    def _run_validation_epoch(self):
        self._do_on_validation_epoch_start()
        self.model.eval()

        with torch.no_grad():
            for batch in self.val_dataloader:
                self._do_on_validation_step_start()
                result = self.validation_step(batch)
                if result:
                    self.validation_step_results_for_an_epoch.append(result)
                self._do_on_validation_step_end()

        self._do_on_validation_epoch_end()

    def _do_on_validation_epoch_start(self):
        self._set_ddp_barrier()
        self.validation_step_results_for_an_epoch = list()
        self.logger.debug(
            {
                "msg": "Validation started",
                "train_epoch": self.current_train_epoch,
                "train_step": self.current_train_step,
            }
        )
        self.do_on_validation_epoch_start()

    def do_on_validation_epoch_start(self):
        pass

    def _do_on_validation_epoch_end(self):
        self._set_ddp_barrier()
        self.do_on_validation_epoch_end()

    def do_on_validation_epoch_end(self):
        pass

    def train(self):
        for _ in range(self.current_train_epoch + 1, self.train_epochs + 1):
            self._run_train_epoch()
        self._delete_model_snapshot()

    def save_model_state_dicts(
        self,
        is_snapshot: bool = False,
        is_best: bool = False,
    ):
        assert self.is_master_process, "Only save model state dicts in master process"
        state = {
            "epoch": self.current_train_epoch,
            "step": self.current_train_step,
            "model": (
                self.model.module.state_dict()
                if self.is_distributed_training
                else self.model.state_dict()
            ),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }
        if is_best:
            self.logger.log(
                {
                    "msg": "Best model found. Saving model",
                    "train_epoch": self.current_train_epoch,
                    "train_step": self.current_train_step,
                }
            )
            path = os.path.join(self.model_checkpoint_dir, "best.t7")
        elif is_snapshot:
            path = os.path.join(self.model_snapshot_path)
        else:
            self.logger.debug(
                {
                    "msg": "Saving model checkpoint",
                    "train_epoch": self.current_train_epoch,
                    "train_step": self.current_train_step,
                }
            )
            path = os.path.join(
                self.model_checkpoint_dir, f"checkpoint_{self.current_train_step}.t7"
            )
        torch.save(state, path)

    def save_best_model_state_dicts(self):
        self.save_model_state_dicts(is_best=True)

    def save_model_snapshot(self):
        self.save_model_state_dicts(is_snapshot=True)
