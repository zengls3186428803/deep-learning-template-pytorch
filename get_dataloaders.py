import random

import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from myDataset import AvilaDataset, BanknoteAuthentication, SensorDataset


def get_avila_loaders_5(batch_size=128, test_batch_size=1024):
    dataset = AvilaDataset(is_train=True)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    dataset = AvilaDataset(is_train=True)
    train_eval_loader = DataLoader(dataset=dataset, batch_size=test_batch_size, shuffle=False)
    dataset = AvilaDataset(is_train=False)
    test_loader = DataLoader(dataset=dataset, batch_size=test_batch_size, shuffle=False)
    dataset = AvilaDataset(is_train=True)
    dataloader_one = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    dataset = AvilaDataset(is_train=True)
    dataloader_full = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

    return train_loader, train_eval_loader, test_loader, dataloader_one, dataloader_full


def get_fashion_mnist_loaders_5(data_aug=False, batch_size=128, test_batch_size=1000):
    if data_aug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True, num_workers=2, drop_last=True
    )
    dataset = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform_test)
    train_eval_loader = DataLoader(
        dataset,
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )
    dataset = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(
        dataset,
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )
    dataset = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform_test)
    dataloader_one = DataLoader(
        dataset,
        batch_size=1, shuffle=False, num_workers=2, drop_last=True
    )
    dataset = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform_test)
    dataloader_full = DataLoader(
        dataset,
        batch_size=1, shuffle=False, num_workers=2, drop_last=True
    )
    return train_loader, train_eval_loader, test_loader, dataloader_one, dataloader_full


def get_line_loader(batch_size=128, test_batch_size=1000, k=1.5, b=7):
    total = 1000
    data_list = list()
    for i in range(1, total + 1):
        x = random.uniform(0, 100)
        y = k * x + b

        data_list.append([x, y])
    data_set = torch.Tensor(data_list)

    print(data_set)


def get_fashion_mnist_loaders(data_aug=False, batch_size=128, test_batch_size=1000):
    if data_aug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True, num_workers=2, drop_last=True
    )
    dataset = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform_test)
    train_eval_loader = DataLoader(
        dataset,
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )
    dataset = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(
        dataset,
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    return train_loader, train_eval_loader, test_loader


def get_mnist_loaders(data_aug=False, batch_size=128, test_batch_size=1000):
    if data_aug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True, num_workers=2, drop_last=True
    )
    dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform_test)
    train_eval_loader = DataLoader(
        dataset,
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )
    dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(
        dataset,
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    return train_loader, train_eval_loader, test_loader


if __name__ == "__main__":
    # get_mnist_loaders()
    get_line_loader()
    pass
