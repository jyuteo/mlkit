# Description   : An example for training a CNN model using the MLKit framework
# Usage         : [CUDA_VISIBLE_DEVICES=0,1](optional) torchrun --standalone --nproc_per_node=2 test.py  # noqa: E501

import json
import hydra
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


class TrainCNN(Trainer):
    def __init__(
        self,
        cfg: DictConfig,
        **kwargs,
    ):
        super().__init__(
            train_epochs=cfg.train_epochs,
            dataloader_batch_size=cfg.dataloader.batch_size,
            dataloader_num_workers=cfg.dataloader.num_workers,
            learning_rate=cfg.learning_rate,
            step_by_epoch=cfg.step_by_epoch,
            checkpoint_every=cfg.model_checkpoint.checkpoint_every,
            validate_every=cfg.validate_every,
            optimizer_config=cfg.optimizer,
            lr_scheduler_config=cfg.lr_scheduler,
            experiment_log_dir=cfg.log.experiment_log_dir,
            metrics_log_filepath=cfg.log.metrics_log_path,
            model_checkpoint_dir=cfg.model_checkpoint.save_dir,
            model_snapshot_path=cfg.model_snapshot.save_path,
            resume_training=cfg.resume_training.enabled,
            resume_training_model_state_dict_path=cfg.resume_training.model_state_dict_path,
        )

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

    def do_on_train_step_end(self) -> None:
        self.metrics_logger.log(
            "train",
            {
                "train_step": self.current_train_step,
                "lr": self.optimizer.param_groups[0]["lr"],
                **self.train_step_results,
            },
        )

    def do_on_validation_epoch_end(self) -> None:
        self.logger.debug("Validation done. Calculating validation metrics")

        total_loss = 0.0
        total_correct = 0.0
        total_samples = 0

        for step_result in self.validation_step_results_for_an_epoch:
            target = step_result["label"]
            output = step_result["output"]
            pred = step_result["pred"]

            loss = F.cross_entropy(output, target)
            total_loss += loss.item() * target.size(0)
            total_samples += target.size(0)
            total_correct += torch.sum(pred == target)

        epoch_loss = total_loss / total_samples
        epoch_accuracy = total_correct.item() / total_samples

        results = {
            "train_epoch": self.current_train_epoch,
            "train_step": self.current_train_step,
            "loss": epoch_loss,
            "accuracy": epoch_accuracy,
        }
        self.logger.log({"msg": "Validation results", **results})
        self.metrics_logger.log("val", results)

        if self.is_best_model(results):
            self.save_best_model_state_dicts()

    def is_best_model(self, metrics: Dict) -> bool:
        if not self.best_metrics:
            self.best_metrics = metrics
            return False
        return metrics["accuracy"] > self.best_metrics["accuracy"]


def plot_metrics(log_filepath: str):
    with open(log_filepath, "r") as f:
        data = json.load(f)

    train_loss_x, train_loss_y, lr_y = [], [], []
    if "train" in data:
        train_loss_x = [x["train_step"] for x in data["train"]]
        train_loss_y = [x["loss"] for x in data["train"]]
        lr_y = [x["lr"] for x in data["train"]]

    val_loss_x, val_loss_y, val_accuracy_y = [], [], []
    if "val" in data:
        val_loss_x = [x["train_step"] for x in data["val"]]
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
    plt.show()


@hydra.main(config_path="config", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    TrainerUtils.set_random_seed_and_torch_deterministic(**cfg.deterministic)

    if DDPUtils.is_cuda_available():
        DDPUtils.setup_ddp_torchrun()
        trainer = TrainCNN(cfg)
        trainer.train()
        DDPUtils.cleanup_ddp()
    else:
        trainer = TrainCNN(cfg)
        trainer.train()

    plot_metrics(cfg.log.metrics_log_path)


if __name__ == "__main__":
    main()
