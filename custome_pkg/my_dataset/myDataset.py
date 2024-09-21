from torch.utils.data.dataset import Dataset
import torch
import pandas as pd
import numpy as np


def transform_label_to_integer(df, label_col_index):
    labels = df.iloc[:, label_col_index]
    label_set = set()
    for e in labels:
        label_set.add(e)
    label_to_integer = dict()
    i = 0
    for e in label_set:
        label_to_integer[e] = i
        i += 1
    for i in range(0, len(labels)):
        df.iloc[i, label_col_index] = label_to_integer[df.iloc[i, label_col_index]]


def transform_dataframe_to_tensor(df, shuffle=True) -> torch.Tensor:
    torch.manual_seed(0)
    ar = np.array(df).astype(float)
    x = torch.from_numpy(ar)
    x.to(torch.float)
    if shuffle:
        random_indices = torch.randperm(len(x))
        x = x[random_indices]
    return x


class AvilaDataset(Dataset):
    def __init__(self, path="./data/avila", is_train=True, label_col_index=10):
        if is_train:
            path += "/avila-tr.txt"
        else:
            path += "/avila-ts.txt"
        super().__init__()
        self.is_train = is_train
        df = pd.read_csv(path, header=None, names=list(range(0, 11)), index_col=None)
        transform_label_to_integer(df, label_col_index)
        x = transform_dataframe_to_tensor(df)
        self.x = dict()
        self.y = dict()
        if is_train:
            self.key = "train"
        else:
            self.key = "test"

        self.x[self.key] = x[:, :-1].float()
        self.y[self.key] = x[:, -1].to(torch.long)
        # print(torch.unique(self.y[self.key]))

    def __len__(self):
        return len(self.y[self.key])

    def __getitem__(self, item):
        return self.x[self.key][item], self.y[self.key][item]


class BanknoteAuthentication(Dataset):
    def __init__(self, path="./data/banknote", is_train=True,
                 label_col_index=4, proportion=0.8):
        super().__init__()
        path += "/data_banknote_authentication.txt"
        self.is_train = is_train
        df = pd.read_csv(path, header=None, names=list(range(0, 5)), index_col=None)
        transform_label_to_integer(df, label_col_index)
        x = transform_dataframe_to_tensor(df)
        self.x = dict()
        self.y = dict()
        split_index = int(len(x) * proportion)
        if is_train:
            self.key = "train"
            self.x[self.key] = x[:split_index, :-1]
            self.y[self.key] = x[:split_index, -1].to(torch.long)
        else:
            self.key = "test"
            self.x[self.key] = x[split_index:, :-1]
            self.y[self.key] = x[split_index:, -1].to(torch.long)

    def __len__(self):
        return len(self.y[self.key])

    def __getitem__(self, item):
        return self.x[self.key], self.y[self.key]


class SensorDataset(Dataset):
    def __init__(self, path="./data/sensorless/Sensorless_drive_diagnosis.txt",
                 is_train=True,
                 label_col_index=48, proportion=0.8):
        super().__init__()
        self.is_train = is_train
        df = pd.read_csv(path, delimiter=" ", header=None, names=list(range(0, 49)), index_col=None)
        transform_label_to_integer(df, label_col_index)
        x = transform_dataframe_to_tensor(df)
        self.x = dict()
        self.y = dict()
        split_index = int(len(x) * proportion)
        if is_train:
            self.key = "train"
            self.x[self.key] = x[:split_index, :-1]
            self.y[self.key] = x[:split_index, -1].to(torch.long)
        else:
            self.key = "test"
            self.x[self.key] = x[split_index:, :-1]
            self.y[self.key] = x[split_index:, -1].to(torch.long)

    def __len__(self):
        return len(self.y[self.key])

    def __getitem__(self, item):
        return self.x[self.key], self.y[self.key]


class MyDataset(Dataset):
    def __init__(self, num_sample=30, dim=1024):
        super().__init__()
        self.dim = dim
        self.data = dict()
        self.data["x"] = torch.randn(num_sample, dim)
        self.data["y"] = torch.randn(num_sample, dim)

    def __getitem__(self, item):
        return self.data["x"][item], self.data["y"][item]

    def __len__(self):
        return len(self.data["y"])


if __name__ == "__main__":
    print(AvilaDataset().__len__())
    print(BanknoteAuthentication().__len__())
    print(SensorDataset().__len__())
