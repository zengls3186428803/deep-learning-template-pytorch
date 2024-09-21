from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split

from .nature_language import (
    load_alpaca_gpt4,
    load_codefeedback,
    load_gsm8k,
    load_meta_math,
    load_sst2,
    load_wizardlm,
)
from .MT19937 import load_mt19937
from .linear_congruence import load_lcg


def load_mt19937_8bits(seed):
    train_set, validation_set, test_set = load_mt19937(
        max_seq_len=256,
        num_samples=256,
        eval_split_ratio=0.0,
        seed=seed,
        delimiter=",",
        fixed_len=True,
        num_bits=8,
    )
    return train_set, validation_set, test_set


def load_mt19937_12bits(seed):
    train_set, validation_set, test_set = load_mt19937(
        max_seq_len=4096,
        num_samples=4096,
        eval_split_ratio=0.0,
        seed=seed,
        delimiter=",",
        fixed_len=True,
        num_bits=12,
    )
    return train_set, validation_set, test_set


def get_fashion_mnist_dataset():
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    return training_data, test_data, test_data


def get_cifar_10_dataset():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="data", train=True, download=True, transform=transform
    )

    testset = torchvision.datasets.CIFAR10(
        root="data", train=False, download=True, transform=transform
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )
    return trainset, test_set, testset


def get_diagonal_dataset(num_samples=5000, test_ratio=0.2):
    from my_model import DiagonalNeuralNetworkDataset

    dataset = DiagonalNeuralNetworkDataset(num_samples=num_samples)
    train_size = int((1 - test_ratio) * num_samples)
    test_size = num_samples - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(
        f"total_size = {len(dataset)}, train_size, test_size = {len(train_dataset)}, {len(test_dataset)}"
    )
    return train_dataset, test_set, test_dataset


def get_iris_dataset(test_ratio=0.2):
    from my_dataset.iris import IrisDataset

    dataset = IrisDataset()
    num_samples = len(dataset)
    train_size = int((1 - test_ratio) * num_samples)
    test_size = num_samples - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(
        f"total_size = {len(dataset)}, train_size, test_size = {len(train_dataset)}, {len(test_dataset)}"
    )
    return train_dataset, test_set, test_dataset


# return (train_set, validation_set, test_set)
DATASETMAPPING = {
    # language dataset
    "alpaca_gpt4": load_alpaca_gpt4,
    "codefeedback": load_codefeedback,
    "gsm8k": load_gsm8k,
    "meta_math": load_meta_math,
    "sst2": load_sst2,
    "wizardlm": load_wizardlm,
    "mt19937": load_mt19937,
    "lcg": load_lcg,
    "mt19937-8": load_mt19937_8bits,
    "mt19937-12": load_mt19937_12bits,
    # ===========================================
    "FashionMNIST": get_fashion_mnist_dataset,
    "CIFAR10": get_cifar_10_dataset,
    "Diag": get_diagonal_dataset,
    "iris": get_iris_dataset,
}


if __name__ == "__main__":
    train_set, test_set = get_iris_dataset(test_ratio=0.3)
    print(test_set[1])
