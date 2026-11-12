"""Load the pinned Qwen3-0.6B checkpoint for the functional backend and the reference runner."""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"


def load_json(name: str) -> dict:
    with open(CONFIG_DIR / name) as f:
        return json.load(f)


def model_config() -> dict:
    return load_json("model.json")


def load_model():
    """Return (model, tokenizer) in FP32 eval mode with eager attention.

    Loads only from the local Hugging Face cache, at the pinned revision.
    """
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()
    hf_logging.set_verbosity_error()

    cfg = model_config()
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_id"], revision=cfg["tokenizer_revision"], local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        revision=cfg["revision"],
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation=cfg["attention"],
    )
    model.eval()
    check_shapes(model, cfg)
    return model, tokenizer


def check_shapes(model, cfg: dict) -> None:
    c = model.config
    expected = {
        "num_hidden_layers": cfg["num_hidden_layers"],
        "hidden_size": cfg["hidden_size"],
        "intermediate_size": cfg["intermediate_size"],
        "num_attention_heads": cfg["num_attention_heads"],
        "num_key_value_heads": cfg["num_key_value_heads"],
        "head_dim": cfg["head_dim"],
        "vocab_size": cfg["vocab_size"],
        "tie_word_embeddings": cfg["tie_word_embeddings"],
    }
    for key, want in expected.items():
        got = getattr(c, key)
        if got != want:
            raise ValueError(f"checkpoint config {key} is {got}, expected {want}")
    if c._attn_implementation != cfg["attention"]:
        raise ValueError(f"attention backend is {c._attn_implementation}, expected {cfg['attention']}")

    h, d = cfg["hidden_size"], cfg["head_dim"]
    nq, nkv, ff = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["intermediate_size"]
    attn = model.model.layers[0].self_attn
    mlp = model.model.layers[0].mlp
    shapes = {
        "q_proj": (attn.q_proj.weight.shape, (nq * d, h)),
        "k_proj": (attn.k_proj.weight.shape, (nkv * d, h)),
        "v_proj": (attn.v_proj.weight.shape, (nkv * d, h)),
        "o_proj": (attn.o_proj.weight.shape, (h, nq * d)),
        "q_norm": (attn.q_norm.weight.shape, (d,)),
        "k_norm": (attn.k_norm.weight.shape, (d,)),
        "gate_proj": (mlp.gate_proj.weight.shape, (ff, h)),
        "up_proj": (mlp.up_proj.weight.shape, (ff, h)),
        "down_proj": (mlp.down_proj.weight.shape, (h, ff)),
        "embed_tokens": (model.model.embed_tokens.weight.shape, (cfg["vocab_size"], h)),
    }
    for name, (got, want) in shapes.items():
        if tuple(got) != want:
            raise ValueError(f"{name} weight shape {tuple(got)}, expected {want}")
    # The checkpoint stores lm_head.weight and embed_tokens.weight separately,
    # so transformers 5.0 loads two tensors instead of sharing one. Tied here
    # means equal values.
    import torch

    if not torch.equal(model.lm_head.weight, model.model.embed_tokens.weight):
        raise ValueError("lm_head weight differs from embed_tokens weight")
