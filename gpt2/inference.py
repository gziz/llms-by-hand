import torch

def generate_text_simple(model, idx, max_new_tokens, context_size):

    for _ in range(max_new_tokens):
        idx_curr_window = idx[:, -context_size:] # 1
        with torch.no_grad():
            logits = model(idx_curr_window)

        logits = logits[:, -1, :] # 2
        probs = torch.softmax(logits, dim=-1) # 3
        next_idx = torch.argmax(probs, dim=-1, keepdim=True)

        idx = torch.cat((idx, next_idx), dim=-1)

    return idx
