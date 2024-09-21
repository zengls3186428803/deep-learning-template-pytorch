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
    x = x.to(torch.float)
    if shuffle:
        random_indices = torch.randperm(len(x))
        x = x[random_indices]
    return x


class IrisDataset(Dataset):
    def __init__(self, path="./data/Iris.csv", label_col_index=5):
        super().__init__()
        df = pd.read_csv(path, header=None, index_col=None)
        df = df.drop(index=0).reset_index(drop=True)
        transform_label_to_integer(df, label_col_index)
        x = transform_dataframe_to_tensor(df)
        self.x = x[:, 1:-1]
        self.y = x[:, -1].to(torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return self.x[item], self.y[item]


if __name__ == "__main__":
    IrisDataset()
