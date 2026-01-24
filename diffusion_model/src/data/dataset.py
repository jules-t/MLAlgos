import torch
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.transforms import v2


def get_data(path):
    transform = v2.Compose([
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ])

    data = datasets.CIFAR10(
                root=path,
                train=True,
                transform=transform,
                download=True
                )
    return data


def get_dataloader(path, batch_size):
    data = get_data(path)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    return dataloader