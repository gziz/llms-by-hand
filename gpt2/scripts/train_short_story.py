import urllib

import tiktoken
import torch
from huggingface_hub import hf_hub_download

from gpt2.configs import model_configs
from gpt2.model import GPTModel
from gpt2.train import train_model_simple
from gpt2.utils import create_dataloader_v1

gpt2_small_cfg = model_configs["gpt2-small"]
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load pretrained HF GPT-2 state dict
sd_hf = torch.load(
    hf_hub_download(repo_id="gpt2", filename="pytorch_model.bin"),
    map_location="cpu",
)

model = GPTModel(gpt2_small_cfg)
# load_weights_into_gpt(model, sd_hf)
model.to(device)

tokenizer = tiktoken.get_encoding("gpt2")


file_path = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/28c65cdfbc3338e2e040016eea4b7fdf556e4d57/ch02/01_main-chapter-code/the-verdict.txt"
with urllib.request.urlopen(file_path) as f:
    text_data = f.read().decode('utf-8')


# Use 90% for training and 10% for validation
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=gpt2_small_cfg["context_length"],
    stride=gpt2_small_cfg["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=gpt2_small_cfg["context_length"],
    stride=gpt2_small_cfg["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
train_model_simple(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    num_epochs=10,
    eval_freq=5,
    eval_iter=10,
    device=device,
    start_context="Every",
    tokenizer=tokenizer,
)