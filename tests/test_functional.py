"""Functional-backend rules, driven in-process without actsim.

Loads the pinned model once (a few seconds on CPU). Requires the reference
capture for case 'capital' (reference/capture.py capital).

Run: .venv/bin/python tests/test_functional.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from runtime import functional as F  # noqa: E402

LAYER = 2
OPS = F.OPS


def cid(op: str, layer: int = LAYER) -> int:
    return layer * 16 + OPS[op]


def step(rt, op, *inputs, layer=LAYER):
    return rt.commit(cid(op, layer), rt.compute(cid(op, layer), tuple(inputs)))


def run_layer(rt) -> int:
    """Drive one block in graph order; returns the block output token."""
    x = rt.source_layer_input(LAYER)
    rt.fork(x, 2)
    t = step(rt, "input_norm", x)
    for op in ("qkv_projection", "qk_norm_rope", "cache_update", "attention", "attention_output_projection"):
        t = step(rt, op, t)
    r1 = step(rt, "attention_residual_add", t, x)
    rt.fork(r1, 2)
    m = step(rt, "mlp", step(rt, "post_attention_norm", r1))
    return step(rt, "mlp_residual_add", m, r1)


def fresh(model_from=None):
    rt = F.Runtime(run_id="test", run_dir=Path(tempfile.mkdtemp()), case="capital")
    if model_from is not None:
        rt.model, rt.tokenizer = model_from.model, model_from.tokenizer
    rt.ensure_reference()
    rt.ensure_model()
    return rt


def expect_error(fn, needle):
    try:
        fn()
    except F.BridgeError as exc:
        assert needle in str(exc), f"expected {needle!r} in {exc}"
        return
    raise AssertionError(f"expected BridgeError containing {needle!r}")


def main():
    torch.set_grad_enabled(False)
    base = fresh()

    # The layer matches, and every artifact was consumed exactly as registered.
    out = run_layer(base)
    rows = base.compare()["rows"]
    assert rows and all(r["ok"] for r in rows), [r for r in rows if not r["ok"]]
    assert all(a.consumers_remaining == 0 for a in base.artifacts[1:] if a and a.token_id != out)
    print(f"layer {LAYER}: {len(rows)} comparisons ok")

    # The comparison has teeth: a 1e-3 perturbation of one committed tensor is caught.
    rt = fresh(base)
    run_layer(rt)
    mlp = next(a for a in rt.artifacts[1:] if a and a.producer.endswith(".mlp"))
    rt.tensors[mlp.tensors["hidden"]] += 1e-3
    bad = [r for r in rt.compare()["rows"] if not r["ok"]]
    assert [r["producer"] for r in bad] == [mlp.producer], bad
    print(f"perturbation detected at {bad[0]['producer']} (tolerance ratio {bad[0]['tolerance_ratio']:.1f})")

    # A second append of the same positions is rejected at commit.
    rt = fresh(base)
    x = rt.source_layer_input(LAYER)
    rt.fork(x, 2)
    heads = step(rt, "qk_norm_rope", step(rt, "qkv_projection", step(rt, "input_norm", x)))
    rt.artifact(heads).consumers_remaining = 2
    step(rt, "cache_update", heads)
    h = rt.compute(cid("cache_update"), (heads,))
    expect_error(lambda: rt.commit(cid("cache_update"), h), "duplicate or missing append")
    print("duplicate cache append rejected")

    # Repeated delivery of a token beyond its registered consumers is rejected.
    expect_error(lambda: rt.compute(cid("cache_update"), (heads,)), "no consumers left")
    print("repeated delivery rejected")

    # A join of operands from different invocations is rejected.
    rt = fresh(base)
    a = rt.source_layer_input(LAYER)
    b = rt.source_layer_input(LAYER)
    other = rt.artifact(b)
    other.identity = F.Identity(other.identity.run_id, other.identity.request_id, 1, "decode",
                                other.identity.position_start, other.identity.sequence_length)
    expect_error(lambda: rt.compute(cid("attention_residual_add"), (a, b)), "join identity mismatch")
    print("mismatched join rejected")

    # Attention refuses a query whose cache version is stale.
    rt = fresh(base)
    x = rt.source_layer_input(LAYER)
    rt.fork(x, 2)
    t = step(rt, "input_norm", x)
    for op in ("qkv_projection", "qk_norm_rope", "cache_update"):
        t = step(rt, op, t)
    rt.kv[("test", 1, LAYER)].version += 1
    expect_error(lambda: rt.compute(cid("attention"), (t,)), "stale cache version")
    print("stale cache version rejected")

    # Missing keys and commits by the wrong component fail with context.
    expect_error(lambda: rt.artifact(9999), "missing token id 9999")
    h = rt.compute(cid("input_norm"), (rt.source_layer_input(LAYER),))
    expect_error(lambda: rt.commit(cid("mlp"), h), "belongs to component")
    print("missing token and wrong-component commit rejected")
    print("functional tests passed")


if __name__ == "__main__":
    main()
