from typing import Optional

import tiktoken
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from gpt2.inference import generate
from gpt2.utils import text_to_token_ids, token_ids_to_text


def calc_loss_batch(
    input_batch: Tensor, target_batch: Tensor, model: Module, device: str
):
    "Given an input and target batch, run the model and get the loss."
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


# Given a dataloader (e.g. training, validation) get the loss for num_batches of the dataloader
def calc_loss_loader(
    data_loader: DataLoader,
    model: Module,
    device: str,
    num_batches: Optional[int] = None,
):
    """
    Given a dataloader (e.g. training, validation) get the loss for num_batches of the dataloader.
    If num_batches is None, will calculate loss on the full dataloader.
    """
    model.eval()
    total_loss = 0
    if len(data_loader) == 0:
        return float(0)

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        total_loss += calc_loss_batch(
            input_batch, target_batch, model=model, device=device
        )

    return total_loss / num_batches


def generate_and_print_sample(
    model: Module, tokenizer: tiktoken.Encoding, device: str, start_context: str
):
    "Function used to print generated text across checkpoints of the model throughout training"
    model.eval()
    base_model = model.module if hasattr(model, "module") else model
    context_size = base_model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size,
            temperature=0.8,
            top_k=40,
        )

    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()


def evaluate_model(
    model: Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    eval_iter: int,
):
    """
    Evaluate the model using the training and validation dataloaders.
    Note: eval_iter is used to determine how many batches are used, you're not evaluating on the full train dataset!
    """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
        model.train()

    return train_loss, val_loss


def train_model_simple(
    model: Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    device: str,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
    tokenizer: tiktoken.Encoding,
    scheduler=None,
    generate_freq=None,
):
    "A simple training loop for a GPT model. Evaluates the model every eval_freq steps and prints the train and val loss."
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        for i, (input_batch, target_batch) in enumerate(train_loader):
            optimizer.zero_grad()  # 2A
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )  # 2F

            if generate_freq is not None and global_step % generate_freq == 0:
                generate_and_print_sample(model, tokenizer, device, start_context)

        if generate_freq is None:
            generate_and_print_sample(model, tokenizer, device, start_context)  # 3

    return train_losses, val_losses, track_tokens_seen
