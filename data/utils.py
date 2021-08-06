import torch
from torch.utils.data.distributed import DistributedSampler

def get_dataloader(dataset, batch_size, num_workers, 
        train=True, distributed=False, world_size=1):

    if train:
        drop_last = True
        shuffle = True
        sampler = torch.utils.data.RandomSampler(dataset)
    else:
        drop_last = False
        shuffle = False
        sampler = torch.utils.data.SequentialSampler(dataset)

    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        batch_size = batch_size // world_size

    dataloader = torch.utils.data.DataLoader(dataset, 
            batch_size=batch_size,
            sampler=sampler,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=True)

    return dataloader

