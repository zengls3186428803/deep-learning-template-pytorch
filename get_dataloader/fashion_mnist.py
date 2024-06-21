from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def get_fashion_mnist_loaders_5(batch_size=128, test_batch_size=1000):
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
