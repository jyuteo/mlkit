import json
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from typing import Dict
from omegaconf import DictConfig
from torchvision import datasets, transforms

from mlkit.base import Trainer
from mlkit.utils import set_random_seed_and_torch_deterministic


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
            dataloader_shuffle=cfg.dataloader.shuffle,
            learning_rate=cfg.learning_rate,
            step_by_epoch=cfg.step_by_epoch,
            checkpoint_every=cfg.checkpoint_every,
            validate_every=cfg.validate_every,
            optimizer_config=cfg.optimizer,
            lr_scheduler_config=cfg.lr_scheduler,
            experiment_log_filepath=cfg.log_filepath.experiment,
            metrics_log_filepath=cfg.log_filepath.metrics,
        )

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


def plot_metrics(log_filepath: str):
    with open(log_filepath, "r") as f:
        data = json.load(f)

    train_loss_x = [x["train_step"] for x in data["train"]]
    train_loss_y = [x["loss"] for x in data["train"]]
    lr_y = [x["lr"] for x in data["train"]]

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
    set_random_seed_and_torch_deterministic(**cfg.deterministic)
    trainer = TrainCNN(cfg)
    trainer.train()
    plot_metrics(cfg.log_filepath.metrics)


if __name__ == "__main__":
    main()
