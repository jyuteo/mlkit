# Description   : An example for training a CNN model using the MLKit framework
# Usage         : [CUDA_VISIBLE_DEVICES=0,1](optional) torchrun --standalone --nproc_per_node=2 train_example.py  # noqa: E501

import os
import hydra
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from typing import Dict, Tuple
from omegaconf import DictConfig
from torchvision import datasets, transforms

from mlkit.trainer import Trainer
from mlkit.utils.trainer_utils import TrainerUtils
from mlkit.utils.ddp_utils import DDPUtils
from mlkit.configs import (
    LogConfig,
    WandBConfig,
    TrainConfig,
    DataLoaderConfig,
    OptimizerConfig,
    LRSchedulerConfig,
    ModelCheckpointConfig,
    ModelSnapshotConfig,
)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.softmax(x, dim=1)
        return output


class CNNTrainer(Trainer):
    def __init__(
        self,
        config: TrainConfig,
        **kwargs,
    ):
        super().__init__(config)

        self.best_metrics: Dict = {}

    def build_model(self) -> torch.nn.Module:
        return Net()

    def build_dataset(
        self,
    ) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_dataset = datasets.MNIST(
            "../data", train=True, download=True, transform=transform
        )
        val_dataset = datasets.MNIST("../data", train=False, transform=transform)
        return train_dataset, val_dataset

    def train_step(self, batch_data) -> Dict:
        data, target = batch_data
        output = self.model(data)
        loss = F.cross_entropy(output, target)
        return {
            "loss": loss,
        }

    def validation_step(self, batch_data) -> Dict:
        data, target = batch_data
        output = self.model(data)
        pred = output.argmax(dim=1, keepdim=False)
        return {
            "label": target,
            "output": output,
            "pred": pred,
        }

    def do_on_validation_epoch_end(self) -> None:
        self.logger.debug("Validation done. Calculating validation metrics")

        label = [
            step_output["label"]
            for step_output in self.validation_step_results_for_an_epoch
        ]
        output = [
            step_output["output"]
            for step_output in self.validation_step_results_for_an_epoch
        ]
        pred = [
            step_output["pred"]
            for step_output in self.validation_step_results_for_an_epoch
        ]

        label = torch.cat(label, dim=0)
        output = torch.cat(output, dim=0)
        pred = torch.cat(pred, dim=0)

        label = DDPUtils.all_gather_tensors(label)
        output = DDPUtils.all_gather_tensors(output)
        pred = DDPUtils.all_gather_tensors(pred)

        total_loss = F.cross_entropy(output, label).item() * label.size(0)
        total_correct = torch.sum(pred == label).item()
        total_data = label.size(0)

        epoch_loss = total_loss / total_data
        epoch_accuracy = total_correct / total_data

        results = {
            "loss": epoch_loss,
            "accuracy": epoch_accuracy,
        }

        if self.is_best_model(results):
            self.save_best_model()

        self.logger.info({"msg": "Validation results", **results})
        self.metrics_logger.log(results, self.current_train_step, "val")
        self.log_wandb_metrics(results, "val")

    def is_best_model(self, metrics: Dict) -> bool:
        if not self.best_metrics:
            self.best_metrics = metrics
            return False
        else:
            if metrics["accuracy"] > self.best_metrics["accuracy"]:
                self.best_metrics = metrics
                return True
            else:
                return False


@hydra.main(config_path="config", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    TrainerUtils.set_random_seed_and_torch_deterministic(**cfg.deterministic)

    train_config = TrainConfig(
        env_vars_file_path=cfg.env_vars_file_path,
        train_epochs=cfg.train_epochs,
        step_by_epoch=cfg.step_by_epoch,
        validate_every=cfg.validate_every,
        learning_rate=cfg.learning_rate,
        log=LogConfig(**cfg.log),
        wandb=WandBConfig(**cfg.wandb),
        dataloader=DataLoaderConfig(**cfg.dataloader),
        optimizer=OptimizerConfig(**cfg.optimizer),
        lr_scheduler=LRSchedulerConfig(**cfg.lr_scheduler),
        model_checkpoint=ModelCheckpointConfig(**cfg.model_checkpoint),
        model_snapshot=ModelSnapshotConfig(**cfg.model_snapshot),
    )

    if DDPUtils.is_cuda_available():
        DDPUtils.setup_ddp_torchrun()
        trainer = CNNTrainer(train_config)
        trainer.train()
        DDPUtils.cleanup_ddp()
    else:
        trainer = CNNTrainer(train_config)
        trainer.train()


if __name__ == "__main__":
    main()
