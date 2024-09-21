import datasets
from tqdm import tqdm
from my_utils.decorator import cache_to_disk


@cache_to_disk()
def load_meta_math(max_tokens=666, num_samples=100000, eval_split_ratio=0.1, seed=31):
    # total 395000 samples
    train_size = num_samples - eval_split_ratio * num_samples
    dataset = datasets.load_dataset("meta-math/MetaMathQA", split="train")

    def preprocess(data):
        return {
            "x": f'Q: {data["query"]}\nA: ',
            "y": data["response"].split("\nThe answer is:")[0],
        }

    train_samples = []
    eval_samples = []
    count = 0
    dataset.shuffle(seed=seed)
    bar = tqdm(dataset, total=num_samples)
    total = 0
    ok = 0
    for sample in dataset:
        total += 1
        temp = preprocess(sample)
        if (
                len(temp["x"] + " " + temp["y"]) >= max_tokens
                or "GSM" not in sample["type"]
        ):
            continue
        bar.update(1)
        bar.set_description(f"ok: {ok}/{total}")
        ok += 1
        processed_sample = preprocess(sample)
        if count < train_size:  # First 100,000 samples for training
            train_samples.append(processed_sample)
        elif train_size <= count < num_samples:  # Next 10,000 samples for evaluation
            eval_samples.append(processed_sample)
        elif count >= num_samples:  # Stop processing after collecting enough samples
            break
        count += 1
    train_set = datasets.Dataset.from_list(train_samples)
    eval_set = datasets.Dataset.from_list(eval_samples)
    return train_set, eval_set, eval_set


@cache_to_disk()
def load_gsm8k():
    dataset = datasets.load_dataset("gsm8k", "main")
    dataset = dataset.map(
        lambda e: {
            "x": f'Q: {e["question"]}\nA: ',
            "y": e["answer"],
        }
    )
    train_set = dataset["train"]
    test_set = dataset["test"]
    return train_set, test_set, test_set


@cache_to_disk()
def load_sst2():
    dataset = datasets.load_dataset("glue", "sst2")
    instruction = "classify the sentiment of the text: "
    label_map = {0: "negative", 1: "positive", -1: "other"}
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["sentence"]}\nresult: ',
            "y": label_map[e["label"]],
        }
    )
    train_set = dataset["train"]
    validation_set = dataset["validation"]
    test_set = dataset["test"]
    return train_set, validation_set, test_set


@cache_to_disk("data_cache")
def load_codefeedback(max_tokens=512, num_samples=100000, eval_split_ratio=0.1, seed=31):
    template_wo_input = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Response:
    """
    dataset = datasets.load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train")
    train_size = num_samples - eval_split_ratio * num_samples

    def preprocess(data):
        y = data["answer"]
        y = "```".join(y.split("```")[:2]) + "```"  # only keep the first code block
        return {
            "x": template_wo_input.format(instruction=data["query"]),
            "y": y,
        }

    train_samples = []
    eval_samples = []
    count = 0
    dataset.shuffle(seed=seed)
    bar = tqdm(dataset, total=num_samples)
    total = 0
    ok = 0
    for sample in dataset:
        total += 1
        temp = preprocess(sample)
        if "```" not in sample["answer"]:
            continue
        if len(temp["x"] + " " + temp["y"]) >= max_tokens:
            continue
        bar.update(1)
        bar.set_description(f"ok: {ok}/{total}")
        ok += 1
        processed_sample = preprocess(sample)
        if count < train_size:
            train_samples.append(processed_sample)
        elif train_size <= count < num_samples:
            eval_samples.append(processed_sample)
        elif count >= num_samples:  # Stop processing after collecting enough samples
            break
        count += 1

    train_set = datasets.Dataset.from_list(train_samples)
    eval_set = datasets.Dataset.from_list(eval_samples)
    return train_set, eval_set, eval_set


@cache_to_disk("data_cache")
def load_wizardlm(max_tokens=512, num_samples=70000, eval_split_ratio=0.1, seed=31):
    template_wo_input = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Response:
    """
    dataset = datasets.load_dataset("silk-road/Wizard-LM-Chinese-instruct-evol", split="train")
    train_size = num_samples - eval_split_ratio * num_samples

    def preprocess(data):
        y = data["output"]
        return {
            "x": template_wo_input.format(instruction=data["instruction"]),
            "y": y,
        }

    train_samples = []
    eval_samples = []
    count = 0
    dataset.shuffle(seed=seed)
    bar = tqdm(dataset, total=num_samples)
    total = 0
    ok = 0
    for sample in dataset:
        total += 1
        temp = preprocess(sample)
        if "sorry" in temp["y"].lower() or "as an ai" in temp["y"].lower():
            continue
        if len(temp["x"] + " " + temp["y"]) >= max_tokens:
            continue
        bar.update(1)
        bar.set_description(f"ok: {ok}/{total}")
        ok += 1
        processed_sample = temp
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


@cache_to_disk("data_cache")
def load_alpaca_gpt4():
    template_wo_input = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Response:
    """
    template_with_input = """### Instruction:
    {instruction}

    ### Input:
    {input}

    ### Response:
    """

    dataset = datasets.load_dataset("tatsu-lab/alpaca")

    def alpaca_preprocess(instruction, input, output):
        if input == "":
            x = template_wo_input.format(instruction=instruction)
        else:
            x = template_with_input.format(instruction=instruction, input=input)
        return {"x": x, "y": output}

    dataset = dataset.map(
        lambda e: alpaca_preprocess(e["instruction"], e["input"], e["output"])
    )
    # we sample 10% of the training set as validation set
    train_set = dataset["train"].train_test_split(test_size=0.1)["train"]
    validation_set = dataset["train"].train_test_split(test_size=0.1)["test"]
    return train_set, validation_set, validation_set


DATASET_MAP = {
    "sst2": load_sst2,
    "gsm8k": load_gsm8k,
    "meta_math": load_meta_math,
    "codefeedback": load_codefeedback,
    "wizard_lm": load_wizardlm,
    "alpaca_gpt4": load_alpaca_gpt4,
}

if __name__ == '__main__':
    load_meta_math(max_tokens=512, num_samples=100000, eval_split_ratio=0.1, seed=31)
