import os
import random
import torch
import numpy as np


class TrainerUtils:
    @staticmethod
    def set_random_seed_and_torch_deterministic(
        random_seed: int,
        torch_use_deterministic_algorithms: bool = True,
        cudnn_backend_deterministic: bool = True,
        **kwargs,
    ):
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        os.environ["PYTHONHASHSEED"] = str(random_seed)
        print(f"Random seed set as {random_seed}")

        if torch_use_deterministic_algorithms:
            assert cudnn_backend_deterministic, (
                "If torch_use_deterministic_algorithms is enabled, "
                "cudnn_backend_deterministic mode should also be enabled"
            )
            torch.use_deterministic_algorithms(True)
            print("PyTorch use deterministic algorithms enabled")
        elif cudnn_backend_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print("CuDNN backend set to deterministic mode")
