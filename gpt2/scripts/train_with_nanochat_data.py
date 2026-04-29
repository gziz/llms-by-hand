from gpt2.train import train_model_simple
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from gpt2.model import GPTModel
from gpt2.configs import model_configs
from gpt2.utils import text_to_token_ids, token_ids_to_text, create_dataloader_v2
from nanochat.dataset import parquets_iter_batched

import tiktoken

# Load data from nanochat parquet shards
MAX_TRAIN_DOCS = 100_000
MAX_VAL_DOCS = 2000

train_texts = []
for batch in parquets_iter_batched(split="train"):
    print(f"Loaded batch of {len(batch)} training documents")
    train_texts.extend(batch)
    if len(train_texts) >= MAX_TRAIN_DOCS:
        train_texts = train_texts[:MAX_TRAIN_DOCS]
        break

val_texts = []
for batch in parquets_iter_batched(split="val"):
    print(f"Loaded batch of {len(batch)} validation documents")
    val_texts.extend(batch)
    if len(val_texts) >= MAX_VAL_DOCS:
        val_texts = val_texts[:MAX_VAL_DOCS]
        break

# Create data loaders
gpt2_small_cfg = model_configs["gpt2-small"]
train_loader = create_dataloader_v2(
    train_texts, batch_size=64,
    max_length=gpt2_small_cfg["context_length"],
    stride=128, shuffle=True
)
val_loader = create_dataloader_v2(
    val_texts, batch_size=64,
    max_length=gpt2_small_cfg["context_length"],
    stride=128, shuffle=False
)


# SETUP MODEL
device = "cuda" if torch.cuda.is_available() else "cpu"

model = GPTModel(gpt2_small_cfg)
model.to(device)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
    model = torch.nn.DataParallel(model)

num_epochs = 2
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader))
tokenizer = tiktoken.get_encoding("gpt2")

# RUN TRAINING JOB
train_model_simple(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=num_epochs,
    eval_freq=5,
    eval_iter=10,
    device=device,
    start_context="Once upon",
    tokenizer=tokenizer,
    generate_freq=50,
)