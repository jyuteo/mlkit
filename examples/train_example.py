# Description   : An example for training a CNN model using the MLKit framework
# Usage         : [CUDA_VISIBLE_DEVICES=0,1](optional) torchrun --standalone --nproc_per_node=2 train_example.py  # noqa: E501

import os
import hydra
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from typing import Dict
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
        train_config: TrainConfig,
        **kwargs,
    ):
        super().__init__(train_config)

        self.best_metrics: Dict = {}

    def build_model(self) -> torch.nn.Module:
        return Net()

    def build_dataset(self) -> torch.utils.data.Dataset:
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
            self.save_best_model_state_dicts()

        self.logger.log({"msg": "Validation results", **results})
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


def plot_metrics(log_filepath: str):
    with open(log_filepath, "r") as f:
        data = json.load(f)

    train_loss_x, train_loss_y, lr_y = [], [], []
    if "train" in data:
        train_loss_x = [x["step"] for x in data["train"]]
        train_loss_y = [x["loss"] for x in data["train"]]
        lr_y = [x["lr"] for x in data["train"]]

    val_loss_x, val_loss_y, val_accuracy_y = [], [], []
    if "val" in data:
        val_loss_x = [x["step"] for x in data["val"]]
        val_loss_y = [x["loss"] for x in data["val"]]
        val_accuracy_y = [x["accuracy"] for x in data["val"]]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    ax1.plot(train_loss_x, train_loss_y, label="train", color="blue")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_title("Training Loss")

    ax2.plot(val_loss_x, val_loss_y, label="val", color="orange")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.set_title("Validation Loss")

    ax3.plot(val_loss_x, val_accuracy_y, label="val", color="green")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Accuracy")
    ax3.legend()
    ax3.set_title("Validation Accuracy")

    ax4.plot(lr_y, label="lr", color="yellow")
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Learning Rate")
    ax4.legend()
    ax4.set_title("Learning Rate")

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(log_filepath), "metrics.png"))


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

    plot_metrics(cfg.log.metrics_log_path)


if __name__ == "__main__":
    main()
