import os
import torch
import torch.distributed as dist

def _get_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))

# def _is_local():
#     return get_rank() == 0

def _get_world_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE", "1"))

rank: int
world_size: int

initialized : bool = False

def get_rank() -> int:
    return rank

def get_world_size() -> int:
    return world_size

def init_dist() -> int:
    global rank
    global world_size
    global initialized
    rank = _get_rank()
    world_size = _get_world_size()
    # if world_size < 2:
    #     raise ValueError("Too few gpus to parallelize")
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    initialized = True
    return rank

def shard(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.size(dim=dim) % world_size == 0
    return torch.tensor_split(x, world_size, dim=dim)[rank]



