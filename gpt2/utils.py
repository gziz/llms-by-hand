import torch
import torch.nn as nn
import numpy as np

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(
            f"Shape mismatch. Left: {left.shape}, Right: {right.shape}"
        )
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    "gpt is our model, params is the dictionary of pretrained weights"

    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe']) # WPE
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte']) # WTE
    
    for b in range(len(params["blocks"])):
        # Wq, Wk, Wv
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].attn.Wq.weight = assign(
            gpt.trf_blocks[b].attn.Wq.weight, q_w.T)
        gpt.trf_blocks[b].attn.Wk.weight = assign(
            gpt.trf_blocks[b].attn.Wk.weight, k_w.T)
        gpt.trf_blocks[b].attn.Wv.weight = assign(
            gpt.trf_blocks[b].attn.Wv.weight, v_w.T)
        # Bias for Wq, Wk, Wv
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].attn.Wq.bias = assign(
            gpt.trf_blocks[b].attn.Wq.bias, q_b)
        gpt.trf_blocks[b].attn.Wk.bias = assign(
            gpt.trf_blocks[b].attn.Wk.bias, k_b)
        gpt.trf_blocks[b].attn.Wv.bias = assign(
            gpt.trf_blocks[b].attn.Wv.bias, v_b)

        # OUT PROJECTION
        gpt.trf_blocks[b].attn.out_proj.weight = assign(
            gpt.trf_blocks[b].attn.out_proj.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].attn.out_proj.bias = assign(
            gpt.trf_blocks[b].attn.out_proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])
        
        # FFN (l1 weights, l1 bias, l2 weights, l2 bias)
        gpt.trf_blocks[b].ffn.layers[0].weight = assign(
            gpt.trf_blocks[b].ffn.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ffn.layers[0].bias = assign(
            gpt.trf_blocks[b].ffn.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ffn.layers[2].weight = assign(
            gpt.trf_blocks[b].ffn.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ffn.layers[2].bias = assign(
            gpt.trf_blocks[b].ffn.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])
        # Layer norms
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
