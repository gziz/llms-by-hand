import torch
import torch.nn as nn

import tiktoken

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    return torch.tensor(encoded).unsqueeze(0)

def token_ids_to_text(tokens, tokenizer):
    flat = tokens.squeeze(0)
    return tokenizer.decode(flat.tolist())

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(
            f"Shape mismatch. Left: {left.shape}, Right: {right.shape}"
        )
    if isinstance(right, torch.Tensor):
        return torch.nn.Parameter(right.detach().clone())
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    """Load a Hugging Face GPT-2 state_dict (e.g. from `pytorch_model.bin`) into our model.

    HF GPT-2 stores `c_attn`, `c_proj`, `c_fc` as Conv1D layers whose weight
    shape is (in_features, out_features), which is transposed relative to
    `nn.Linear` (out_features, in_features). So those weights are `.T`'d.
    The LM head weight is tied to the token embedding (`wte.weight`).
    """

    # Embeddings
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte.weight"])
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe.weight"])

    n_blocks = len(gpt.trf_blocks)

    for b in range(n_blocks):
        prefix = f"h.{b}"

        # --- Attention: fused QKV. Conv1D weight is [emb, 3*emb] ---
        c_attn_w = params[f"{prefix}.attn.c_attn.weight"]
        c_attn_b = params[f"{prefix}.attn.c_attn.bias"]
        q_w, k_w, v_w = c_attn_w.chunk(3, dim=-1)
        q_b, k_b, v_b = c_attn_b.chunk(3, dim=-1)

        gpt.trf_blocks[b].attn.Wq.weight = assign(
            gpt.trf_blocks[b].attn.Wq.weight, q_w.T)
        gpt.trf_blocks[b].attn.Wk.weight = assign(
            gpt.trf_blocks[b].attn.Wk.weight, k_w.T)
        gpt.trf_blocks[b].attn.Wv.weight = assign(
            gpt.trf_blocks[b].attn.Wv.weight, v_w.T)

        gpt.trf_blocks[b].attn.Wq.bias = assign(
            gpt.trf_blocks[b].attn.Wq.bias, q_b)
        gpt.trf_blocks[b].attn.Wk.bias = assign(
            gpt.trf_blocks[b].attn.Wk.bias, k_b)
        gpt.trf_blocks[b].attn.Wv.bias = assign(
            gpt.trf_blocks[b].attn.Wv.bias, v_b)

        # --- Attention output projection ---
        gpt.trf_blocks[b].attn.out_proj.weight = assign(
            gpt.trf_blocks[b].attn.out_proj.weight,
            params[f"{prefix}.attn.c_proj.weight"].T)
        gpt.trf_blocks[b].attn.out_proj.bias = assign(
            gpt.trf_blocks[b].attn.out_proj.bias,
            params[f"{prefix}.attn.c_proj.bias"])

        # --- Feed-forward (layers[0] = c_fc, layers[2] = c_proj) ---
        gpt.trf_blocks[b].ffn.layers[0].weight = assign(
            gpt.trf_blocks[b].ffn.layers[0].weight,
            params[f"{prefix}.mlp.c_fc.weight"].T)
        gpt.trf_blocks[b].ffn.layers[0].bias = assign(
            gpt.trf_blocks[b].ffn.layers[0].bias,
            params[f"{prefix}.mlp.c_fc.bias"])
        gpt.trf_blocks[b].ffn.layers[2].weight = assign(
            gpt.trf_blocks[b].ffn.layers[2].weight,
            params[f"{prefix}.mlp.c_proj.weight"].T)
        gpt.trf_blocks[b].ffn.layers[2].bias = assign(
            gpt.trf_blocks[b].ffn.layers[2].bias,
            params[f"{prefix}.mlp.c_proj.bias"])

        # --- LayerNorms inside the block ---
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, params[f"{prefix}.ln_1.weight"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, params[f"{prefix}.ln_1.bias"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, params[f"{prefix}.ln_2.weight"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, params[f"{prefix}.ln_2.bias"])

    # Final LayerNorm
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["ln_f.weight"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["ln_f.bias"])

    # LM head: weight is tied to token embedding in GPT-2
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte.weight"])
