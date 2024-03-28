import os
import torch

from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Tuple, Any
from pathlib import Path

from .logger import Logger
from .metrics_logger import MetricsLogger


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
        dataloader_shuffle: bool = True,
        step_by_epoch: bool = True,
        validate_every: int = 1,
        experiment_log_filepath: str = "./logs/log.json",
        metrics_log_filepath: str = "./logs/metrics_log.json",
        model_checkpoint_dir: str = "./checkpoints",
        **kwargs,
    ):
        self.train_epochs = train_epochs
        self.step_by_epoch = step_by_epoch
        self.checkpoint_every = checkpoint_every
        self.validate_every = validate_every

        self.batch_size = dataloader_batch_size
        self.num_workers = dataloader_num_workers
        self.dataloader_shuffle = dataloader_shuffle

        self.train_dataset, self.val_dataset = self.build_dataset()
        self.train_dataloader, self.val_dataloader = self.build_dataloader()

        self.model = self.build_model()

        self.learning_rate = learning_rate
        self.optimizer = self.build_optimizer(**optimizer_config)
        self.lr_scheduler = self.build_lr_scheduler(**lr_scheduler_config)

        self.current_train_epoch = 0
        self.current_train_step = 0

        self.train_step_results: Dict = dict()
        self.validation_step_results_for_an_epoch: List[Dict] = list()

        self.logger = Logger(experiment_log_filepath)
        self.metrics_logger = MetricsLogger(metrics_log_filepath)

        self.model_checkpoint_dir = model_checkpoint_dir
        if not os.path.exists(self.model_checkpoint_dir):
            dir = Path(self.model_checkpoint_dir)
            dir.mkdir(parents=True, exist_ok=True)

        config = locals()
        config.pop("self")
        self.logger.log({"msg": "Config", **config})

    def build_dataset(self) -> Tuple[Dataset, Dataset]:
        raise NotImplementedError

    def build_dataloader(self) -> Tuple[DataLoader, DataLoader]:
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.dataloader_shuffle,
        )
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.dataloader_shuffle,
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
        step_size: int = 30,
        gamma: float = 0.1,
        **kwargs,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size, gamma)

    def train_step(self, batch_data: Any) -> Dict:
        raise NotImplementedError

    def _do_on_train_step_start(self):
        self.current_train_step += 1
        self.train_step_results = dict()
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
        if not self.step_by_epoch:
            if self.current_train_step % self.checkpoint_every == 0:
                self.save_model_checkpoint()
            if self.current_train_step % self.validate_every == 0:
                self._run_validation_epoch()
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
        self.do_on_train_epoch_start()

    def do_on_train_epoch_start(self):
        pass

    def _do_on_train_epoch_end(self):
        if self.step_by_epoch:
            if self.current_train_epoch % self.checkpoint_every == 0:
                self.save_model_checkpoint()
            if self.current_train_epoch % self.validate_every == 0:
                self._run_validation_epoch()
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
        self.do_on_validation_epoch_end()

    def do_on_validation_epoch_end(self):
        pass

    def train(self):
        for _ in range(self.current_train_epoch + 1, self.train_epochs + 1):
            self._run_train_epoch()

    def save_model_checkpoint(
        self,
        is_best: bool = False,
    ):
        state = {
            "epoch": self.current_train_epoch,
            "step": self.current_train_step,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
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
        else:
            self.logger.log(
                {
                    "msg": "Saving model checkpoint",
                    "train_epoch": self.current_train_epoch,
                    "train_step": self.current_train_step,
                }
            )
            path = os.path.join(self.model_checkpoint_dir, "checkpoint.t7")
        torch.save(state, path)

    def save_best_model_checkpoint(self):
        self.save_model_checkpoint(is_best=True)
