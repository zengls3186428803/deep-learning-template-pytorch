import copy
import os
import collections
from typing import Iterable, List
from tqdm import tqdm

# os.environ["http_proxy"] = "http://127.0.0.1:10808"
# os.environ["https_proxy"] = "http://127.0.0.1:10808"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset, DownloadConfig


class GPT2Tokenizer:
    def __init__(self):
        self.special_vocab = None
        self.inv_vocab = None
        self.vocab = None
        self.has_add_special_tokens = False
        self.has_build_vocab = False
        self.default_special_token_list = ["[BOS]", "[EOS]", "[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"]
        self.eos_token = None
        self.set_eos_token("[EOS]")

    def set_eos_token(self, val: str = "[EOS]"):
        self.eos_token = val

    def tokenize(self, text_list: List[str], with_eos=True):
        assert self.has_build_vocab, "haven't build vocab, please call <build_vocab> method fist! "
        ids_list = list()
        for text in text_list:
            tmp_list = list()
            for word in text:
                tmp_list.append(
                    self.vocab.get(word, self.vocab.get("UNK", 0))
                )
            ids_list.append(tmp_list)
        return ids_list

    def build_vocab(self, text_list: list, max_vocab_size=10000, min_freq=1):
        counter = collections.Counter()
        p_bar = tqdm(total=len(text_list), desc="counting token in texts")
        for text in text_list:
            tokens = list(text)
            counter.update(tokens)
            p_bar.update(1)

        if not self.has_add_special_tokens:
            self.add_special_tokens(self.default_special_token_list)
        vocab = copy.deepcopy(self.special_vocab)
        p_bar = tqdm(total=max(counter.total(), max_vocab_size), desc="specifying id to tokens")
        for token, freq in counter.most_common(max_vocab_size):
            if freq >= min_freq and token not in vocab:
                vocab[token] = len(vocab)
            p_bar.update(1)

        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.has_build_vocab = True

    def add_special_tokens(self, special_token_list: List[str]):
        self.special_vocab = {special_token_list[i]: i for i in range(len(special_token_list))}
        self.has_add_special_tokens = True

    def save_state_dict(self, save_directory="model_pretrained/gpt2"):
        import json
        state_dict = self.__dict__
        state_dict["vocab"] = self.vocab
        with open(f"{save_directory}/vocab.json", "w") as f:
            json.dump(self.__dict__, f)

    def load_state_dict(self, save_directory="model_pretrained/gpt2"):
        import json
        with open(f"{save_directory}/vocab.json", "r") as f:
            state_dict = json.load(f)
            self.__dict__ = state_dict

    def ids_to_tensor(self, ids_list: List[List[int]]):
        for ids in ids_list:
            pass




if __name__ == "__main__":
    # repo_path = "liwu/MNBVC"
    # subset_name = 'law_judgement'
    # split = "train"
    # dataset = load_dataset(
    #     path=repo_path,
    #     name='law_judgement',
    #     split='train',
    #     streaming=True,
    #     download_config=DownloadConfig(
    #         cache_dir="data/.huggingface" + "/" + repo_path,
    #     ),
    #     trust_remote_code=True,
    # )
    # dataset.save_to_disk("data/" + repo_path + "/" + subset_name)

    tokenizer = GPT2Tokenizer()
    tokenizer.build_vocab([data["text"] for data in [{"text": "我真今天啊啊啊777起其1的爱你YYYs啊"}]])
    ids = tokenizer.tokenize(["今天天机七七i七七i其"])
    print(tokenizer.vocab)
    print(ids)
    print(tokenizer.__dict__)
    tokenizer.save_state_dict()
    tokenizer = GPT2Tokenizer()
    tokenizer.load_state_dict()
    print(tokenizer.vocab)
