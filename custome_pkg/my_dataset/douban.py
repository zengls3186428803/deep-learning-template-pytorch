
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class DouBan(Dataset):
    def __init__(self, datapath: str, train: bool = True, test_proportion: float = 0.2):
        super().__init__()
        data = pd.read_csv(datapath, sep='\t', header=None, names=['sentence'])
        train_data, test_data = train_test_split(data, test_size=test_proportion, random_state=42)
        if train:
            self.data = train_data
        else:
            self.data = test_data
        self.data.reset_index(drop=True, inplace=True)

    def __getitem__(self, index):
        # 返回单条数据（句子）
        return self.data.iloc[index]['sentence'].replace(" ", "")

    def __len__(self):
        # 返回数据集长度
        return len(self.data)

