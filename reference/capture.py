"""Pinned reference run: capture every per-layer intermediate of an unmodified Qwen3 forward.

Writes runs/reference/<case>/capture.pt and manifest.json. The ACT functional
backend compares its committed artifacts against capture.pt.

Usage:
    .venv/bin/python reference/capture.py [case ...]
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import transformers  # noqa: E402
from transformers.models.qwen3 import modeling_qwen3  # noqa: E402

from runtime.model import load_json, load_model, model_config  # noqa: E402

RUNS = ROOT / "runs" / "reference"


def _cpu(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to("cpu", torch.float32).clone()


class Recorder:
    """Forward hooks plus a wrapper around the module-level eager attention function.

    Qwen3Attention looks up `eager_attention_forward` in its module globals at
    call time, so wrapping that global records post-RoPE Q, the cache K/V the
    attention actually read, and the pre-o_proj output, while the computation
    itself is still the pinned eager implementation.
    """

    def __init__(self, model, tensors: dict[str, torch.Tensor], prefix: str = ""):
        self.model = model
        self.tensors = tensors
        self.prefix = prefix
        self.handles = []
        self._orig_attn = modeling_qwen3.eager_attention_forward

    def _put(self, key: str, t: torch.Tensor) -> None:
        key = self.prefix + key
        if key in self.tensors:
            raise RuntimeError(f"reference key {key} captured twice")
        self.tensors[key] = _cpu(t)

    def _out(self, module, key: str):
        # A forward hook that returns a value replaces the module output, so this returns None.
        def hook(mod, args, out):
            self._put(key, out[0] if isinstance(out, tuple) else out)

        self.handles.append(module.register_forward_hook(hook))

    def _in(self, module, key: str):
        def hook(mod, args, kwargs):
            self._put(key, args[0] if args else kwargs["hidden_states"])

        self.handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))

    def __enter__(self):
        m = self.model.model
        self._out(m.embed_tokens, "embed")
        for i, layer in enumerate(m.layers):
            p = f"layers.{i}."
            attn = layer.self_attn
            self._in(layer, p + "input")
            self._out(layer, p + "layer_out")
            self._out(layer.input_layernorm, p + "input_norm")
            self._out(attn.q_proj, p + "q_proj")
            self._out(attn.k_proj, p + "k_proj")
            self._out(attn.v_proj, p + "v_proj")
            self._in(attn.o_proj, p + "attn_core")
            self._out(attn.o_proj, p + "attn_out")
            self._in(layer.post_attention_layernorm, p + "resid1")
            self._out(layer.post_attention_layernorm, p + "post_norm")
            self._out(layer.mlp, p + "mlp_out")
        self._out(m.norm, "final_norm")

        orig = self._orig_attn

        def wrapped(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            p = f"layers.{module.layer_idx}."
            self._put(p + "q_rope", query)
            self._put(p + "k_cache", key)
            self._put(p + "v_cache", value)
            return orig(module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs)

        modeling_qwen3.eager_attention_forward = wrapped
        return self

    def __exit__(self, *exc):
        modeling_qwen3.eager_attention_forward = self._orig_attn
        for h in self.handles:
            h.remove()
        self.handles.clear()


def git_rev(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def capture_case(model, tokenizer, case: dict) -> dict:
    ids = tokenizer(case["prompt"], return_tensors="pt", add_special_tokens=False).input_ids
    out: dict = {"prompt": case["prompt"], "input_ids": ids.clone()}

    tensors: dict[str, torch.Tensor] = {}
    out["tensors"] = tensors
    with torch.no_grad():
        with Recorder(model, tensors):
            prefill = model(input_ids=ids, use_cache=True)
        logits = _cpu(prefill.logits)
        tensors["logits_last"] = logits[:, -1, :]

        # Cached greedy decode, each step checked against a full-prefix recomputation.
        # Decode invocation k (1-based) is recorded under the prefix "decode<k>.".
        cache = prefill.past_key_values
        seq = ids.clone()
        next_tok = int(logits[0, -1].argmax())
        steps = []
        for step in range(case["decode_steps"]):
            seq = torch.cat([seq, torch.tensor([[next_tok]])], dim=1)
            pos = seq.shape[1] - 1
            prefix = f"decode{step + 1}."
            with Recorder(model, tensors, prefix):
                cached = model(
                    input_ids=torch.tensor([[next_tok]]),
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=torch.tensor([pos]),
                )
            cache = cached.past_key_values
            full = model(input_ids=seq, use_cache=False)
            lc = _cpu(cached.logits[:, -1, :])
            lf = _cpu(full.logits[:, -1, :])
            tensors[prefix + "logits_last"] = lc
            tensors[prefix + "logits_full_prefix"] = lf
            steps.append({
                "position": pos,
                "input_token": next_tok,
                "logits_cached": lc,
                "logits_full_prefix": lf,
                "max_abs_cached_vs_full": float((lc - lf).abs().max()),
                "greedy_cached": int(lc.argmax()),
                "greedy_full_prefix": int(lf.argmax()),
            })
            next_tok = int(lc.argmax())
        out["decode"] = steps
        out["generated_ids"] = [s["greedy_cached"] for s in steps]
        out["first_generated_id"] = int(logits[0, -1].argmax())
    return out


def manifest(case: dict, cap: dict, tokenizer) -> dict:
    cfg = model_config()
    toolchain = load_json("toolchain.json")
    first = cap["first_generated_id"]
    return {
        "case": case["name"],
        "created_unix": time.time(),
        "model_id": cfg["model_id"],
        "checkpoint_revision": cfg["revision"],
        "tokenizer_revision": cfg["tokenizer_revision"],
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "device": "cpu",
        "dtype": "float32",
        "attention_backend": cfg["attention"],
        "generation": {"do_sample": False, "chat_template": cfg["chat_template"], "decode_steps": case["decode_steps"]},
        "act_toolchain": toolchain["repos"],
        "prompt": case["prompt"],
        "input_ids": cap["input_ids"][0].tolist(),
        "prompt_length": int(cap["input_ids"].shape[1]),
        "greedy_ids": [first] + cap["generated_ids"],
        "greedy_text": tokenizer.decode([first] + cap["generated_ids"]),
        "decode_checks": [
            {k: s[k] for k in ("position", "input_token", "max_abs_cached_vs_full", "greedy_cached", "greedy_full_prefix")}
            for s in cap["decode"]
        ],
        "tensor_key_count": len(cap["tensors"]),
        "tensor_bytes": int(sum(t.numel() * t.element_size() for t in cap["tensors"].values())),
        "project_git_revision": git_rev(ROOT),
    }


def main() -> int:
    cases = {c["name"]: c for c in load_json("prompts.json")["cases"]}
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*", default=list(cases))
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    model, tokenizer = load_model()
    for name in args.cases:
        if name not in cases:
            print(f"unknown case {name}; known: {', '.join(cases)}", file=sys.stderr)
            return 2
        case = cases[name]
        cap = capture_case(model, tokenizer, case)
        out_dir = RUNS / name
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(cap, out_dir / "capture.pt")
        man = manifest(case, cap, tokenizer)
        with open(out_dir / "manifest.json", "w") as f:
            json.dump(man, f, indent=2)
        print(f"{name}: {man['prompt_length']} prompt tokens, greedy {man['greedy_text']!r}, "
              f"{man['tensor_bytes'] / 2**20:.1f} MiB captured -> {out_dir}")
        for s in man["decode_checks"]:
            print(f"  decode pos {s['position']}: cached vs full-prefix max abs {s['max_abs_cached_vs_full']:.3e}, "
                  f"greedy {s['greedy_cached']} / {s['greedy_full_prefix']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
