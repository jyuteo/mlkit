# mlkit

A module for PyTorch ML training, evaluation and inference

## Overview

This project is a modular deep learning training framework implemented in PyTorch. It provides functionalities for training, evaluation, and inference of models, along with logging capabilities using Weights & Biases (WandB). The framework supports distributed training using PyTorch's DistributedDataParallel (DDP) executed with torchrun.

Training, evaluation, and inference can be conducted using a standardized approach, requiring minimal user design.

## Features

- **Modular Design**: The is a modular framework allowing easy extension and customization of various components such as data loading and model architecture
- **Distributed Training**: Supports distributed training across multiple GPUs using PyTorch's DistributedDataParallel (DDP) and torchrun.
- **Experiment Logging**: Provides integration with WandB for experiment tracking and visualization.
- **Configurability**: Configuration files (YAML format) are used to specify training, evaluation, and inference settings.
- **Checkpointing**: Automatically saves model snapshots and checkpoints during training at specified intervals, enabling resume of training from the last checkpoint or snapshot in case of interruptions.

## Setup

1. Create python environment and install requirements
   ```shell
   pip install poetry
   poetry install
   pip install -r requirements.txt
   ```
2. Create a .env file in the project root directory and add the necessary environment variables
   ```shell
   WANDB_HOST=
   WANDB_API_KEY=
   ```

## Installation
To install this package in your python environment, run
```shell
pip install git+ssh://git@github.com/jyuteo/mlkit.git
```

## Usage

Create configuration files (train.yaml, eval.yaml, inference.yaml) specifying the experiment settings, including model, data, optimization, and logging parameters.

Instantiate the Trainer class with the appropriate configuration object (TrainConfig, EvaluationConfig, InferenceConfig) and run the desired job (train, evaluate, or inference).

Define and implement required methods for respective steps

- Training and validation

  ```python
  from mlkit.trainer import Trainer
  from mlkit.configs import TrainConfig

  class CNNTrainer(Trainer):
    def __init__(
        self,
        config: TrainConfig,
        **kwargs,
    ):
        super().__init__(config)

    def build_model(self) -> torch.nn.Module:
        # implement it

    def build_dataset(self) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        # implement it

    def train_step(self, batch_data) -> Dict:
        # implement it

    def validation_step(self, batch_data) -> Dict:
        # implement it

    if __name__ == "__main__":
        config = TrainConfig(...)
        trainer = CNNTrainer(config)
        trainer.train()
  ```

- Evaluation

  ```python
    from mlkit.trainer import Trainer
    from mlkit.configs import EvaluationConfig

    class CNNEvaluator(Trainer):
        def __init__(
            self,
            config: EvaluationConfig,
            **kwargs,
        ):
            super().__init__(config)

        def build_model(self) -> torch.nn.Module:
            # implement it
            # not required when torchscript (.pt) model file is used

        def build_dataset(self) -> Tuple[None, torch.utils.data.Dataset]:
            # implement it

        def validation_step(self, batch_data) -> Dict:
            # implement it

    if __name__ == "__main__":
        config = EvaluationConfig(...)
        trainer = CNNEvaluator(config)
        trainer.evaluate()
  ```

- Inference

  ```python
    from mlkit.trainer import Trainer
    from mlkit.configs import InferenceConfig

    class CNNInference(Trainer):
        def __init__(
            self,
            config: InferenceConfig,
            **kwargs,
        ):
            super().__init__(config)

        def build_model(self) -> torch.nn.Module:
            # implement it
            # not required when torchscript (.pt) model file is used

        def build_dataset(self) -> Tuple[None, torch.utils.data.Dataset]:
            # implement it

        def inference_step(self, batch_data) -> Dict:
            # implement it

    if __name__ == "__main__":
        config = InferenceConfig(...)
        trainer = CNNInference(config)
        trainer.inference()
  ```

## Examples

### Training and validation

- [train config example](examples/config/train.yaml)
- [train example](examples/train_example.py)

### Run evaluation and log metrics

- [evaluation config example](examples/config/evaluation.yaml)
- [evaluation example](examples/evaluation_example.py)

### Run inference and get model prediction

- [inference config example](examples/config/inference.yaml)
- [inference example](examples/inference_example.py)
