import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# 自定义Dataset类
class IrisDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def get_Iris_dataloader(batch_size=16, test_batch_size=16):
    # 加载Iris数据集
    iris = load_iris()
    X = iris.data
    y = iris.target

    # 拆分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 创建Dataset对象
    train_dataset = IrisDataset(X_train, y_train)
    test_dataset = IrisDataset(X_test, y_test)

    # 创建DataLoader对象
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    train_eval_loader = DataLoader(train_dataset, batch_size=test_batch_size, shuffle=False)
    dataloader_one = DataLoader(train_dataset, batch_size=1, shuffle=False)
    dataloader_full = DataLoader(train_dataset, batch_size=1, shuffle=False)

    return train_loader, train_eval_loader, test_loader, dataloader_one, dataloader_full
