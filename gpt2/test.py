
import torch
from huggingface_hub import hf_hub_download

from model import GPTModel
from configs import model_configs
from utils import load_weights_into_gpt, text_to_token_ids, token_ids_to_text
from inference import generate
import tiktoken

gpt2_small_cfg = model_configs["gpt2-small"]
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load pretrained HF GPT-2 state dict
sd_hf = torch.load(
    hf_hub_download(repo_id="gpt2", filename="pytorch_model.bin"),
    map_location="cpu",
)

model = GPTModel(gpt2_small_cfg)
load_weights_into_gpt(model, sd_hf)
model.to(device)

tokenizer = tiktoken.get_encoding("gpt2")

torch.manual_seed(123)
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=15,
    context_size=gpt2_small_cfg["context_length"],
    top_k=25,
    temperature=1.4
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
