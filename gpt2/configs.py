base_config = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True,
}

model_configs = {
    "gpt2-small": {**base_config, "emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium": {**base_config, "emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large": {**base_config, "emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl": {**base_config, "emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
