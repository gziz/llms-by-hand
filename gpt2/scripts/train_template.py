from gpt2.train import train_model_simple
import torch
from gpt2.model import GPTModel
from gpt2.configs import model_configs
from gpt2.utils import text_to_token_ids, token_ids_to_text, create_dataloader_v2
from nanochat.dataset import parquets_iter_batched

import tiktoken


# Create data loaders
### TODO ###
gpt2_small_cfg = model_configs["gpt2-small"]
train_loader = []
val_loader = []

# SETUP MODEL
device = "cuda" if torch.cuda.is_available() else "cpu"

model = GPTModel(gpt2_small_cfg)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
tokenizer = tiktoken.get_encoding("gpt2")

# RUN TRAINING JOB
train_model_simple(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    num_epochs=10,
    eval_freq=5,
    eval_iter=10,
    device=device,
    start_context="Once upon",
    tokenizer=tokenizer,
)