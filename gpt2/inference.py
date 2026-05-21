import torch


def generate_text_simple(model, idx, max_new_tokens, context_size):

    for _ in range(max_new_tokens):
        idx_curr_window = idx[:, -context_size:]  # 1
        with torch.no_grad():
            logits = model(idx_curr_window)

        logits = logits[:, -1, :]  # 2
        probs = torch.softmax(logits, dim=-1)  # 3
        next_idx = torch.argmax(probs, dim=-1, keepdim=True)

        idx = torch.cat((idx, next_idx), dim=-1)

    return idx


# Includes top_k, temperature and EOS break
def generate(
    model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None
):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)
            logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits
            )

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
