import collections
import copy
import pickle
from typing import List

from tqdm import tqdm
import json


class GPT2Tokenizer:
    def __init__(self):
        self.special_vocab = None
        self.inv_vocab = None
        self.vocab = None
        self.has_add_special_tokens = False
        self.has_build_vocab = False
        self.default_special_token_list = ["[BOS]", "[EOS]", "[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"]
        self.eos_token = None
        self.pad_token = None
        self.set_eos_token("[EOS]")
        self.set_pad_token("[PAD]")
        self.eos_id = None,
        self.pad_id = None,

    def set_eos_token(self, val: str = "[EOS]"):
        self.eos_token = val

    def set_pad_token(self, val: str = "[PAD]"):
        self.pad_token = val

    def tokenize(self, text):
        return list(text)

    def convert_token_to_id(self, token_list):
        assert self.has_build_vocab, "haven't build vocab, please call <build_vocab> method fist! "
        id_list = list()
        for token in token_list:
            id_list.append(
                self.vocab.get(token, self.vocab.get("UNK", 0))
            )
        return id_list

    def convert_id_to_token(self, id_list):
        token_list = list()
        for index in id_list:
            token_list.append(
                self.inv_vocab.get(index, "[UNK]")
            )
        return token_list

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
        self.eos_id = vocab[self.eos_token]
        self.pad_id = vocab[self.pad_token]

    def add_special_tokens(self, special_token_list: List[str]):
        self.special_vocab = {special_token_list[i]: i for i in range(len(special_token_list))}
        self.has_add_special_tokens = True

    def save_state_dict(self, save_directory="model_pretrained/gpt2"):
        state_dict = self.__dict__
        state_dict["vocab"] = self.vocab
        with open(f"{save_directory}/tokenizer.pkl", "wb") as f:
            pickle.dump(self.__dict__, f)
        with open(f"{save_directory}/vocab.json", "w") as f:
            json.dump(self.vocab, f)

    def load_state_dict(self, save_directory="model_pretrained/gpt2"):
        with open(f"{save_directory}/tokenizer.pkl", "rb") as f:
            state_dict = pickle.load(f)
            self.__dict__.update(state_dict)

    def get_vocab_size(self):
        return len(self.vocab)
