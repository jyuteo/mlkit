import os
import torch

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Union, Tuple, List


class DDPUtils:
    @staticmethod
    def is_cuda_available():
        """
        Returns whether GPU is available
        """
        if torch.cuda.device_count() > 0:
            assert torch.distributed.is_available()
            return True
        return False

    @staticmethod
    def get_device():
        """
        Returns the device of the current process
        """
        if not DDPUtils.is_cuda_available():
            return "cpu"
        return DDPUtils.get_rank()

    @staticmethod
    def get_rank():
        """
        Returns rank of current process
        """

        if DDPUtils.is_cuda_available():
            return int(os.environ["LOCAL_RANK"])
        return None

    @staticmethod
    def get_world_size():
        """
        Returns the total number of processes
        """
        if DDPUtils.is_cuda_available():
            return torch.distributed.get_world_size()
        return 0

    @staticmethod
    def is_master_process():
        """
        Returns whether current process is the master
        """
        return DDPUtils.get_rank() == 0

    @staticmethod
    def set_barrier():
        """
        Sets barrier for all processes to wait for
        """
        assert DDPUtils.is_cuda_available()
        torch.distributed.barrier()

    @staticmethod
    def setup_ddp_torchrun():
        """
        Sets up Distributed Data Parallel using torchrun
        """
        assert DDPUtils.is_cuda_available()
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(DDPUtils.get_rank())

    @staticmethod
    def cleanup_ddp():
        """
        Cleans up Distributed Data Parallel processes
        """
        assert DDPUtils.is_cuda_available()
        torch.distributed.destroy_process_group()

    @staticmethod
    def move_model_to_device(model: torch.nn.Module) -> torch.nn.Module:
        device = DDPUtils.get_device()
        if device == "cpu":
            return model.to(device)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.to(device)
        model = DDP(model, device_ids=[device])
        return model

    @staticmethod
    def move_tensor_or_tuple_of_tensors_to_device(
        data: Union[torch.Tensor, Tuple[torch.Tensor]]
    ):
        device = DDPUtils.get_device()
        if isinstance(data, (tuple, list)):
            assert all(isinstance(item, torch.Tensor) for item in data)
            return (item.to(device) for item in data)
        if isinstance(data, torch.Tensor):
            return data.to(device)
        raise TypeError(
            f"Expected type torch.Tensor or a tuple of torch.Tensor but got {type(data)}"
        )

    @staticmethod
    def all_gather_tensors(
        tensor: torch.Tensor, sync_grad: bool = False
    ) -> torch.Tensor:
        """
        Gathers all tensors from all processes and concatenates them
        """
        world_size = DDPUtils.get_world_size()
        if world_size <= 1:
            return tensor
        if tensor.ndim == 0:
            tensor = tensor.unsqueeze(0)
        tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(tensor_list, tensor)
        if sync_grad:
            local_rank = DDPUtils.get_rank()
            tensor_list[local_rank] = tensor
        return torch.cat(tensor_list, dim=0)

    @staticmethod
    def all_gather_objects(obj: object) -> List[object]:
        """
        Gathers all objects from all processes and concatenates them
        """
        world_size = DDPUtils.get_world_size()
        if world_size <= 1:
            return [obj]
        obj_list = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(obj_list, obj)
        return obj_list


class NoDuplicateDistributedSampler(DistributedSampler):
    """
    A distributed sampler that doesn't add duplicates.
    Arguments are the same as DistributedSampler
    Refer to https://github.com/pytorch/pytorch/issues/25162#issuecomment-1227647626
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # Some ranks may have fewer samples, that's fine
            if self.rank >= len(self.dataset) % self.num_replicas:
                self.num_samples -= 1
            self.total_size = len(self.dataset)
