# Description   : An example for evaluation of a CNN model using the MLKit framework
# Usage         : [CUDA_VISIBLE_DEVICES=0,1](optional) torchrun --standalone --nproc_per_node=2 evaluation_example.py  # noqa: E501

import hydra
import torch
import torch.nn.functional as F


from typing import Dict, Tuple
from omegaconf import DictConfig
from torchvision import datasets, transforms

from mlkit.trainer import Trainer
from mlkit.utils.trainer_utils import TrainerUtils
from mlkit.utils.ddp_utils import DDPUtils
from mlkit.configs import (
    LogConfig,
    WandBConfig,
    EvaluationConfig,
    DataLoaderConfig,
    ModelFileConfig,
    ModelFileType,
)


class CNNEvaluator(Trainer):
    def __init__(
        self,
        config: EvaluationConfig,
        **kwargs,
    ):
        super().__init__(config)

    def build_dataset(self) -> Tuple[None, torch.utils.data.Dataset]:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        val_dataset = datasets.MNIST("../data", train=False, transform=transform)
        return None, val_dataset

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

        self.logger.info({"msg": "Validation results", **results})
        self.metrics_logger.log(results, self.current_train_step, "val")
        self.log_wandb_metrics(results, "val")


@hydra.main(config_path="config", config_name="evaluation", version_base=None)
def main(cfg: DictConfig) -> None:
    assert cfg.model_file.file_type in (
        "state_dict",
        "torchscript",
    ), "Invalid model_type. Expected 'state_dict' (.t7) or 'torchscript' (.pt)"

    TrainerUtils.set_random_seed_and_torch_deterministic(**cfg.deterministic)

    if cfg.model_file.file_type == "state_dict":
        model_file_type = ModelFileType.STATE_DICT
    else:
        model_file_type = ModelFileType.TORCHSCRIPT

    evaluation_config = EvaluationConfig(
        model_file=ModelFileConfig(
            file_type=model_file_type,
            file_path=cfg.model_file.file_path,
        ),
        env_vars_file_path=cfg.env_vars_file_path,
        log=LogConfig(**cfg.log),
        wandb=WandBConfig(**cfg.wandb),
        dataloader=DataLoaderConfig(**cfg.dataloader),
    )

    if DDPUtils.is_cuda_available():
        DDPUtils.setup_ddp_torchrun()
        trainer = CNNEvaluator(evaluation_config)
        trainer.evaluate()
        DDPUtils.cleanup_ddp()
    else:
        trainer = CNNEvaluator(evaluation_config)
        trainer.evaluate()


if __name__ == "__main__":
    main()
