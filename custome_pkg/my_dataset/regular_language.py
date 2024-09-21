import torch
from torch.utils.data import Dataset, DataLoader
import exrex


class RegularLanguageDataset(Dataset):
    def __init__(self, regex_pattern, max_len=10, data_size=100, limit=100):
        self.regex_pattern = regex_pattern
        generator = exrex.generate(self.regex_pattern, limit=limit)
        self.data = list()
        for s in generator:
            if len(s) <= max_len:
                self.data.append(s)
            if len(self.data) > data_size:
                break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        string = self.data[idx]
        return string


if __name__ == '__main__':
    regex_pattern = r"a*"
    max_len = 20
    dataset_size = 100
    limit = 100
    dataset = RegularLanguageDataset(regex_pattern, max_len=max_len, data_size=dataset_size, limit=limit)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for batch in dataloader:
        print(batch)
