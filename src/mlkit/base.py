import torch

from torch.utils.data import DataLoader, Dataset
from typing import Dict, Tuple, Any

from .utils import checkpoint_model_and_save_current_best


class Trainer:
    def __init__(
        self,
        epochs: int,
        dataloader_batch_size: int,
        dataloader_num_workers: int,
        learning_rate: float,
        checkpoint_every: int,
        optimizer_config: Dict,
        lr_scheduler_config: Dict,
        dataloader_shuffle: bool = True,
        step_by_epoch: bool = True,
        validate_every: int = 1,
        **kwargs,
    ):
        self.train_epochs = epochs
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
        raise NotImplementedError

    def build_lr_scheduler(
        self,
        step: int = 30,
        gamma: float = 0.1,
        **kwargs,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        return torch.optim.lr_scheduler.StepLR(self.optimizer, step, gamma)

    def train_step(self, batch_data: Any) -> Dict:
        raise NotImplementedError

    def _do_on_train_step_start(self):
        if not self.step_by_epoch:
            self.current_train_step += 1
        self.do_on_train_step_start()

    def do_on_train_step_start(self):
        pass

    def _do_on_train_step_end(self):
        if not self.step_by_epoch:
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
        if not self.step_by_epoch:
            checkpoint_model_and_save_current_best(
                self.current_train_step, self.checkpoint_every
            )
        self.do_on_validation_step_end()

    def do_on_validation_step_end(self):
        pass

    def _run_train_epoch(self):
        self._do_on_train_epoch_start()
        self.model.train()

        for batch in self.train_dataloader:
            self._do_on_train_step_start()

            self.optimizer.zero_grad()
            train_metrics_dict = self.train_step(batch)
            train_loss = train_metrics_dict["loss"]
            train_loss.backward()
            self.optimizer.step()

            if not self.step_by_epoch:
                self.lr_scheduler.step()

            self._do_on_train_step_end()

        if self.step_by_epoch:
            self.lr_scheduler.step()

        self._do_on_train_epoch_end()
        # print(f"Epoch {self.current_epoch} training loss: {train_loss.item()}")

    def _do_on_train_epoch_start(self):
        self.current_train_epoch += 1
        print(f"Epoch {self.current_train_epoch}/{self.train_epochs}")
        if self.step_by_epoch:
            self.current_train_step += 1
        self.do_on_train_epoch_start()

    def do_on_train_epoch_start(self):
        pass

    def _do_on_train_epoch_end(self):
        # TODO log metrics
        # TODO checkpoint best model, optimizer, lr_scheduler state dict
        if self.step_by_epoch:
            if self.current_train_step % self.validate_every == 0:
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
                _ = self.validation_step(batch)
                self._do_on_validation_step_end()

        self._do_on_validation_epoch_end()

    def _do_on_validation_epoch_start(self):
        self.do_on_validation_epoch_start()

    def do_on_validation_epoch_start(self):
        pass

    def _do_on_validation_epoch_end(self):
        # TODO log metrics
        # TODO save best model, optimizer, lr_scheduler state dict
        if self.step_by_epoch:
            checkpoint_model_and_save_current_best(
                self.current_train_step, self.checkpoint_every
            )
        self.do_on_validation_epoch_end()

    def do_on_validation_epoch_end(self):
        pass

    def train(self):
        for epoch in range(self.current_train_epoch + 1, self.train_epochs + 1):
            self._run_train_epoch()
