import torch
import pickle
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import random
import numpy as np
import torch.distributed as dist
_LOCAL_PROCESS_GROUP = None


def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD

def _serialize_to_tensor(data, group):
    backend = dist.get_backend(group)
    assert backend in ["gloo", "nccl"]
    device = torch.device("cpu" if backend == "gloo" else "cuda")

    buffer = pickle.dumps(data)
    if len(buffer) > 1024 ** 3:
        print(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                dist.get_rank(), len(buffer) / (1024 ** 3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor

def _pad_to_largest_tensor(tensor, group):
    """
    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    world_size = dist.get_world_size(group=group)
    assert (
            world_size >= 1
    ), "comm.gather/all_gather must be called from ranks within the given group!"
    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)
    size_list = [
        torch.zeros([1], dtype=torch.int64, device=tensor.device) for _ in range(world_size)
    ]
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]

    max_size = max(size_list)

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    if local_size != max_size:
        padding = torch.zeros((max_size - local_size,), dtype=torch.uint8, device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor

def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: list of data gathered from each rank
    """
    if dist.get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return [data]

    tensor = _serialize_to_tensor(data, group)

    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    # receiving Tensor from all ranks
    tensor_list = [
        torch.empty((max_size,), dtype=torch.uint8, device=tensor.device) for _ in size_list
    ]
    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list

def shared_random_seed():
    """
    Returns:
        int: a random number that is the same across all workers.
            If workers need a shared RNG, they can use this shared seed to
            create one.
    All workers must call this function, otherwise it will deadlock.
    """
    ints = np.random.randint(2 ** 31)
    all_ints = all_gather(ints)
    return all_ints[0]

class RandomIdentitySampler_DDP(Sampler):
    """
    Optimized version of RandomIdentitySampler for DDP training.
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        
        # Get world size and rank
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        
        # Calculate per-GPU batch size
        self.mini_batch_size = self.batch_size // self.world_size
        self.num_pids_per_batch = self.mini_batch_size // self.num_instances
        
        # Build index dictionary
        self.index_dic = defaultdict(list)
        for index, (_, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        
        self.pids = list(self.index_dic.keys())
        
        # Precompute the total number of batches
        total_batches = len(self.pids) // self.num_pids_per_batch
        self.batches_per_rank = total_batches // self.world_size
        
        # Calculate length for this rank
        self.length = self.batches_per_rank * self.mini_batch_size
        
        # For reproducibility
        self.epoch = 0
        
    def set_epoch(self, epoch):
        """Set the epoch for reproducibility."""
        self.epoch = epoch
        
    def __iter__(self):
        # Use epoch and rank for deterministic randomness
        seed = self.epoch * 1000 + self.rank
        random.seed(seed)
        np.random.seed(seed)
        
        # Create a list of all pids and shuffle
        all_pids = self.pids.copy()
        random.shuffle(all_pids)
        
        # Calculate how many batches each rank should process
        total_batches = len(all_pids) // self.num_pids_per_batch
        batches_per_rank = total_batches // self.world_size
        
        # Determine the start and end indices for this rank
        start_batch = self.rank * batches_per_rank
        end_batch = start_batch + batches_per_rank
        
        # Generate indices for this rank
        indices = []
        for batch_idx in range(start_batch, end_batch):
            # Get the pids for this batch
            start_pid = batch_idx * self.num_pids_per_batch
            end_pid = start_pid + self.num_pids_per_batch
            batch_pids = all_pids[start_pid:end_pid]
            
            # For each pid, sample instances
            for pid in batch_pids:
                pid_indices = self.index_dic[pid]
                if len(pid_indices) < self.num_instances:
                    # Sample with replacement if not enough instances
                    selected = np.random.choice(pid_indices, self.num_instances, replace=True)
                else:
                    # Sample without replacement if enough instances
                    selected = random.sample(pid_indices, self.num_instances)
                indices.extend(selected)
        
        return iter(indices)
    
    def __len__(self):
        return self.length