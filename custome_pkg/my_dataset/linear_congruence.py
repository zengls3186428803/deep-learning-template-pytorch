import datasets
from tqdm import tqdm
import random
from my_utils.decorator import cache_to_disk


def _int32(x):
    return int(0xFFFFFFFF & x)


class LCG32:
    def __init__(self, seed, a=1664525, c=1013904223, m=2**32):
        self.state = seed
        self.a = a
        self.c = c
        self.m = m

    def extract_number(self):
        self.state = (self.a * self.state + self.c) % self.m
        return _int32(self.state)


def _int8(x):
    return int(0xFF & x)  # 保证结果是8比特


class LCG8:
    def __init__(self, seed, a=97, c=13, m=256):
        self.state = _int8(seed)
        self.a = a
        self.c = c
        self.m = m

    def extract_number(self):
        self.state = _int8(self.a * self.state + self.c) % self.m
        return _int8(self.state)


@cache_to_disk()
def load_lcg(
    max_seq_len=666,
    num_samples=100000,
    eval_split_ratio=0.1,
    seed=31,
    fixed_len=False,
    delimiter=",",
    num_bits=8,
):
    random.seed(seed)
    # total 395000 samples
    train_size = num_samples - eval_split_ratio * num_samples
    sample_list = []
    for i in tqdm(range(num_samples)):
        length = max_seq_len if fixed_len else random.randint(7, max_seq_len)
        seed = random.randint(0, int(1e9))
        if num_bits == 8:
            seed = _int8(seed)
        elif num_bits == 32:
            seed = _int32(seed)
        lcg = LCG32(seed) if num_bits == 32 else LCG8(seed)
        seq = ""
        for j in range(length):
            random_number = lcg.extract_number()
            seq += delimiter + str(random_number)
        sample_list.append({"seed": str(seed), "seq": seq})

    def preprocess(data):
        return {
            "x": data["seed"],
            "y": data["seq"],
        }

    train_samples = []
    eval_samples = []
    count = 0
    bar = tqdm(total=num_samples)
    total = 0
    ok = 0
    for sample in sample_list:
        total += 1
        ok += 1
        bar.set_description(f"ok: {ok}/{total}")
        bar.update(1)
        processed_sample = preprocess(sample)
        if count < train_size:
            train_samples.append(processed_sample)
        elif train_size <= count < num_samples:
            eval_samples.append(processed_sample)
        elif count >= num_samples:
            break
        count += 1
    train_set = datasets.Dataset.from_list(train_samples)
    eval_set = datasets.Dataset.from_list(eval_samples)
    return train_set, eval_set, eval_set


if __name__ == "__main__":
    train_set, eval_set, test_set = load_lcg(
        max_seq_len=256,
        num_samples=256,
        eval_split_ratio=0,
        seed=5,
        delimiter=",",
        num_bits=8,
    )
    print(train_set[13])
