import os

from datasets import load_dataset
from my_utils.decorator import cache_to_disk
from tqdm import tqdm

# os.environ["http_proxy"] = "http://127.0.0.1:10808"
# os.environ["https_proxy"] = "http://127.0.0.1:10808"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def load(repo_path="liwu/MNBVC", subset_name='law_judgement', split="train"):
    @cache_to_disk(root_datadir="data/" + repo_path + "/" + subset_name + "/" + split)
    def load_liwu_law():
        dataset = load_dataset(
            path=repo_path,
            name=subset_name,
            split=split,
            # streaming=True,
            trust_remote_code=True,
        )
        print(next(iter(dataset)))
        data_list = list()
        for data in dataset:
            print(data)
            data_list.append(data)
        print(f"total {len(data_list)}")
        return data_list

    return load_liwu_law()



if __name__ == "__main__":
    load()
