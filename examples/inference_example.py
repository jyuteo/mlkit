# Description   : An example for evaluation of a CNN model using the MLKit framework
# Usage         : [CUDA_VISIBLE_DEVICES=0,1](optional) torchrun --standalone --nproc_per_node=2 inference_example.py  # noqa: E501

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict
from omegaconf import DictConfig
from torchvision import datasets, transforms

from mlkit.trainer import Trainer
from mlkit.utils.trainer_utils import TrainerUtils
from mlkit.utils.ddp_utils import DDPUtils
from mlkit.configs import (
    LogConfig,
    WandBConfig,
    InferenceConfig,
    DataLoaderConfig,
    EvaluationConfig,
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


class CNNInference(Trainer):
    def __init__(
        self,
        config: InferenceConfig,
        **kwargs,
    ):
        super().__init__(config)

    def build_model(self) -> torch.nn.Module:
        return Net()

    def build_dataset(self) -> torch.utils.data.Dataset:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        val_dataset = datasets.MNIST("../data", train=False, transform=transform)
        return None, val_dataset

    def inference_step(self, batch_data) -> Dict:
        data, target = batch_data
        output = self.model(data)
        pred = output.argmax(dim=1, keepdim=False)
        return {
            "label": target,
            "output": output,
            "pred": pred,
        }

    def do_on_inference_epoch_end(self) -> None:
        label = [
            step_output["label"]
            for step_output in self.inference_step_results_for_an_epoch
        ]
        output = [
            step_output["output"]
            for step_output in self.inference_step_results_for_an_epoch
        ]
        pred = [
            step_output["pred"]
            for step_output in self.inference_step_results_for_an_epoch
        ]

        label = torch.cat(label, dim=0)
        output = torch.cat(output, dim=0)
        pred = torch.cat(pred, dim=0)

        label = DDPUtils.all_gather_tensors(label)
        output = DDPUtils.all_gather_tensors(output)
        pred = DDPUtils.all_gather_tensors(pred)

        self.metrics_logger.log(
            {
                "label": label,
                "output": output,
                "pred": pred,
            },
            0,
            "inference",
        )


@hydra.main(config_path="config", config_name="inference", version_base=None)
def main(cfg: DictConfig) -> None:
    TrainerUtils.set_random_seed_and_torch_deterministic(**cfg.deterministic)

    inference_config = EvaluationConfig(
        model_state_dicts_path=cfg.model_state_dicts_path,
        env_vars_file_path=cfg.env_vars_file_path,
        log=LogConfig(**cfg.log),
        wandb=WandBConfig(**cfg.wandb),
        dataloader=DataLoaderConfig(**cfg.dataloader),
    )

    if DDPUtils.is_cuda_available():
        DDPUtils.setup_ddp_torchrun()
        trainer = CNNInference(inference_config)
        trainer.inference()
        DDPUtils.cleanup_ddp()
    else:
        trainer = CNNInference(inference_config)
        trainer.inference()


if __name__ == "__main__":
    main()
