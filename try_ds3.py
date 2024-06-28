import torch
import deepspeed
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


# Sample Dataset
class MyDataset(Dataset):
    def __init__(self, tokenizer, texts, labels, max_len):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


# Sample Data
texts = ["I love machine learning", "Deep learning is amazing"]
labels = [1, 0]

# Parameters
batch_size = 8
max_len = 16
learning_rate = 3e-5
epochs = 3

# Load tokenizer and dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = MyDataset(tokenizer, texts, labels, max_len)
dataloader = DataLoader(dataset, batch_size=batch_size)

# Load model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# DeepSpeed configuration
ds_config = {
    "train_batch_size": batch_size,
    "gradient_accumulation_steps": 1,
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": 5e7,
        "stage3_prefetch_bucket_size": 5e7,
        "stage3_param_persistence_threshold": 1e5,
        "stage3_max_live_parameters": 1e5,
        "stage3_max_reuse_distance": 1e5,
        "stage3_gather_fp16_weights_on_model_save": True
    },
    "steps_per_print": 2000,
    "zero_allow_untested_optimizer": True
}

# Initialize DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    training_data=dataloader,
    config=ds_config
)

# Training loop
for epoch in range(epochs):
    for batch in dataloader:
        input_ids = batch['input_ids'].to(model_engine.local_rank)
        attention_mask = batch['attention_mask'].to(model_engine.local_rank)
        labels = batch['label'].to(model_engine.local_rank)

        outputs = model_engine(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        model_engine.backward(loss)
        model_engine.step()

print("Training complete.")
