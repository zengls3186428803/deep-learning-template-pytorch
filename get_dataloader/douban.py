from torch.utils.data import DataLoader
from class_dataset.douban import DouBan


def get_douban_dataloader(batch_size: int, test_batch_size: int, ):
    datapath = "D:\\code\\py\\Neural_ODE\\data\\chinese-chatbot-corpus-master\\clean_chat_corpus\\douban_single_turn.tsv"
    train_dataset = DouBan(datapath=datapath, train=True)
    eval_dataset = DouBan(datapath=datapath, train=True)
    test_dataset = DouBan(datapath=datapath, train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    dataloader_one = DataLoader(train_dataset, batch_size=1, shuffle=False)

    return train_dataloader, eval_dataloader, test_dataloader, dataloader_one
