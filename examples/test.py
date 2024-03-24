import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra

from typing import Dict
from omegaconf import DictConfig
from torchvision import datasets, transforms

from mlkit.base import Trainer


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
        output = F.log_softmax(x, dim=1)
        return output


class TrainCNN(Trainer):
    def __init__(
        self,
        cfg: DictConfig,
        **kwargs,
    ):
        super().__init__(
            epochs=cfg.epochs,
            dataloader_batch_size=cfg.dataloader.batch_size,
            dataloader_num_workers=cfg.dataloader.num_workers,
            dataloader_shuffle=cfg.dataloader.shuffle,
            learning_rate=cfg.learning_rate,
            step_by_epoch=cfg.step_by_epoch,
            checkpoint_every=cfg.checkpoint_every,
            optimizer_config=cfg.optimizer,
            lr_scheduler_config=cfg.lr_scheduler,
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
        loss = F.nll_loss(output, target)
        return {"loss": loss}

    def validation_step(self, batch_data) -> Dict:
        data, target = batch_data
        output = self.model(data)
        pred = output.argmax(dim=1, keepdim=False)
        return {
            "label": target,
            "output": output,
            "pred": pred,
        }

    def do_on_train_epoch_start(self) -> None:
        print(f"Train epoch {self.current_train_epoch} started")

    def do_on_validation_epoch_start(self) -> None:
        print(f"Validation at train step {self.current_train_step} started")

    def do_on_validation_epoch_end(self) -> None:
        print(f"Validation at train step {self.current_train_step} ended")
        print("Calculating validation metrics")

        total_loss = 0.0
        total_correct = 0.0
        total_samples = 0

        for step_result in self.validation_step_results_for_an_epoch:
            target = step_result["label"]
            output = step_result["output"]
            pred = step_result["pred"]

            loss = F.nll_loss(output, target)

            total_loss += loss.item() * target.size(0)
            total_samples += target.size(0)
            total_correct += torch.sum(pred == target)

        epoch_loss = total_loss / total_samples
        epoch_accuracy = total_correct.item() / total_samples

        print("Validation Loss:", epoch_loss)
        print("Validation Accuracy:", epoch_accuracy)

        # TODO call logger to log metrics"""


@hydra.main(config_path="config", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    print(cfg)
    trainer = TrainCNN(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
