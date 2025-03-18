import os
import torch # type: ignore
import torch.distributed as dist # type: ignore
import hostlist # type: ignore
from .general_utils import Config

global_config = Config()

def setup_for_distributed(is_master):
    """
    Overrides the built-in print function to only allow the master process to print to the console.
    
    Args:
        is_master (bool): Flag indicating if the current process is the master process.
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def is_dist_avail_and_initialized():
    """
    Checks if the distributed computing is available and initialized.
    
    Returns:
        bool: True if the distributed computing is available and initialized; otherwise, False.
    """

    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    """
    Gets the number of processes in the current distributed group.
    
    Returns:
        int: The number of processes in the current group.
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    """
    Gets the rank of the current process in the distributed group.
    
    Returns:
        int: The rank of the current process.
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    """
    Saves a checkpoint only from the master process.
    """
    if is_main_process():
        torch.save(*args, **kwargs)

def init_distributed_mode():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        print('Distributed mode: environment variables RANK and WORLD_SIZE detected.')
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
        print(f'Rank: {rank}, World Size: {world_size}, GPU: {gpu}')
        world_size_ = world_size

    elif all(var in os.environ for var in ["SLURM_PROCID", "SLURM_NTASKS", "SLURM_LOCALID", "SLURM_CPUS_PER_TASK", "SLURM_JOB_NODELIST"]):
        print('Distributed mode: all SLURM environment variables detected.')
        rank = int(os.environ["SLURM_PROCID"])
        gpu = rank % torch.cuda.device_count()
        os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
        
        local_rank = int(os.environ['SLURM_LOCALID'])
        size = int(os.environ['SLURM_NTASKS'])
        cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])
        hostnames = hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST'])
        world_size_ = size

        print(f'Rank: {rank}, GPU: {gpu}, World Size: {world_size_}, CPUs per Task: {cpus_per_task}, Hosts: {hostnames}')

    elif hasattr(global_config, "rank"):
        # Add clear logging or handling if necessary
        print('Distributed mode: using global_config.rank')
        return

    else:
        print("Not running in distributed mode (defaulting to single GPU/CPU).")
        world_size_ = 1
        rank = 0
        gpu = 0

    # Set distributed mode clearly and robustly:
    torch.cuda.set_device(gpu)
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=world_size_,
        rank=rank
    )
    torch.distributed.barrier()
    setup_for_distributed(rank == 0)
    
    return world_size_


def reduce_across_processes(val):
    """
    Reduces a value across all processes so that all processes will have the sum of the value.
    
    Args:
        val (number): The value to be reduced.
    
    Returns:
        torch.Tensor: A tensor with the reduced value.
    """
    if not is_dist_avail_and_initialized():
        # nothing to sync, but we still convert to tensor for consistency with the distributed case.
        return torch.tensor(val)
    t = torch.tensor(val, device="cuda")
    dist.barrier()
    dist.all_reduce(t)
    return t

            


def all_gather_2d(tensor):
    """
    Performs an all-gather operation for a 2D tensor across all processes.
    
    Args:
        tensor (torch.Tensor): The 2D tensor to be gathered.
    
    Returns:
        numpy.ndarray: A numpy array with the gathered tensors from all processes.
    """
    world_size = dist.get_world_size()
    tensor_shape = tensor.shape
    flattened_tensor_shape = (tensor_shape[0] * tensor_shape[1],)
    flattened_tensor = tensor.flatten()
    gathered_tensor_shape = (world_size,) + flattened_tensor_shape
    gathered_tensor = torch.empty(gathered_tensor_shape, dtype=tensor.dtype, device=tensor.device)
    flattened_tensor_list = [torch.empty_like(flattened_tensor) for _ in range(world_size)]
    dist.all_gather(flattened_tensor_list, flattened_tensor)
    for i in range(world_size):
        gathered_tensor[i] = flattened_tensor_list[i]
    output_tensor_shape = (world_size * tensor_shape[0], tensor_shape[1])
    output_tensor = gathered_tensor.view(*output_tensor_shape)
    output_tensor = output_tensor.cpu().detach().numpy()
    return output_tensor


