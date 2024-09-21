import datasets
from tqdm import tqdm
import random
from my_utils.decorator import cache_to_disk


def _int32(x):
    return int(0xFFFFFFFF & x)


class MT19937:
    def __init__(self, seed):
        self.mt = [0] * 624
        self.mt[0] = seed
        self.mti = 0
        for i in range(1, 624):
            self.mt[i] = _int32(
                1812433253 * (self.mt[i - 1] ^ self.mt[i - 1] >> 30) + i
            )

    def extract_number(self):
        if self.mti == 0:
            self.twist()
        y = self.mt[self.mti]
        y = y ^ y >> 11
        y = y ^ y << 7 & 2636928640
        y = y ^ y << 15 & 4022730752
        y = y ^ y >> 18
        self.mti = (self.mti + 1) % 624
        return _int32(y)

    def twist(self):
        for i in range(0, 624):
            y = _int32(
                (self.mt[i] & 0x80000000) + (self.mt[(i + 1) % 624] & 0x7FFFFFFF)
            )
            self.mt[i] = (y >> 1) ^ self.mt[(i + 397) % 624]

            if y % 2 != 0:
                self.mt[i] = self.mt[i] ^ 0x9908B0DF


def _int8(x):
    return int(0xFF & x)  # 保证结果是8比特


class MT8Bit:
    def __init__(self, seed):
        self.mt = [0] * 16  # 使用较小的状态数组
        self.mt[0] = _int8(seed)  # 初始状态的种子
        self.mti = 0
        for i in range(1, 16):
            # 初始化状态数组
            self.mt[i] = _int8(
                (181 * (self.mt[i - 1] ^ (self.mt[i - 1] >> 3)) + i) & 0xFF
            )

    def extract_number(self):
        if self.mti == 0:
            self.twist()
        y = self.mt[self.mti]
        y = y ^ (y >> 3)  # 调整变换
        y = y ^ ((y << 2) & 0xA4)  # 位操作和常数修改
        y = y ^ ((y << 3) & 0xC0)
        y = y ^ (y >> 4)
        self.mti = (self.mti + 1) % 16
        return _int8(y)

    def twist(self):
        for i in range(0, 16):
            y = _int8((self.mt[i] & 0x80) + (self.mt[(i + 1) % 16] & 0x7F))
            self.mt[i] = _int8((y >> 1) ^ self.mt[(i + 13) % 16])  # 修改权重和偏移

            if y % 2 != 0:
                self.mt[i] = self.mt[i] ^ 0x8D  # 使用较小的异或常数


def _int12(x):
    return int(0xFFF & x)  # 保证结果是12比特


class MT12Bit:
    def __init__(self, seed):
        self.mt = [0] * 16  # 使用16个状态值
        self.mt[0] = _int12(seed)  # 初始状态的种子
        self.mti = 0
        for i in range(1, 16):
            # 初始化状态数组
            self.mt[i] = _int12(
                (181 * (self.mt[i - 1] ^ (self.mt[i - 1] >> 3)) + i) & 0xFFF
            )

    def extract_number(self):
        if self.mti == 0:
            self.twist()
        y = self.mt[self.mti]
        y = y ^ (y >> 2)  # 调整变换
        y = y ^ ((y << 5) & 0x1F0)  # 位操作和常数修改
        y = y ^ ((y << 6) & 0x3E0)
        y = y ^ (y >> 10)
        self.mti = (self.mti + 1) % 16
        return _int12(y)

    def twist(self):
        for i in range(16):
            y = _int12((self.mt[i] & 0x800) + (self.mt[(i + 1) % 16] & 0x7FF))
            self.mt[i] = _int12((y >> 1) ^ self.mt[(i + 12) % 16])  # 修改权重和偏移

            if y % 2 != 0:
                self.mt[i] = self.mt[i] ^ 0x8F  # 使用较小的异或常数


@cache_to_disk()
def load_mt19937(
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
        len = max_seq_len if fixed_len else random.randint(7, max_seq_len)
        seed = random.randint(0, int(1e9))
        if num_bits == 8:
            seed = _int8(seed)
        elif num_bits == 32:
            seed = _int32(seed)
        elif num_bits == 12:
            seed = _int12(seed)
        mt = None
        if num_bits == 32:
            mt = MT19937(seed)
        elif num_bits == 8:
            mt = MT8Bit(seed)
        elif num_bits == 12:
            mt = MT12Bit(seed)
        seq = ""
        for j in range(len):
            random_number = mt.extract_number()
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
    train_set, eval_set, test_set = load_mt19937(
        max_seq_len=256,
        num_samples=256,
        eval_split_ratio=0,
        seed=31,
        delimiter=",",
        num_bits=8,
    )
    print(train_set[13])
