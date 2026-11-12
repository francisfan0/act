"""Functional backend: real tensor computation behind ACT token ids.

Protocol for every compute process (brief section 6):
  1. the ACT process receives input token ids from its input memories
  2. compute() resolves them, checks identity and shape, computes into a
     private pending result, and returns a pending handle
  3. commit() publishes the pending result as an immutable artifact with
     lineage and returns its token id
  4. the ACT process sends that token id downstream

Mutable KV state is separate from the artifact table. It changes only at the
cache_update commit, which appends each position exactly once.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from runtime.model import ROOT, load_json, load_model

STAGES = load_json("stage_map.json")
OPS = STAGES["ops"]
TOP_BASE = STAGES["top_level_layer_base"]
TOP_OPS = STAGES["top_level_ops"]
OP_NAMES = {v: k for k, v in OPS.items()}
TOP_NAMES = {v: k for k, v in TOP_OPS.items()}
INVALID = STAGES["invalid_token_id"]
MAX_TOKEN = (1 << STAGES["token_id_width"]) - 1


class BridgeError(Exception):
    pass


@dataclass(frozen=True)
class Identity:
    run_id: str
    request_id: int
    invocation_id: int
    phase: str
    position_start: int
    sequence_length: int

    def join_key(self) -> tuple:
        return (self.run_id, self.request_id, self.invocation_id, self.position_start, self.sequence_length)


@dataclass
class Artifact:
    token_id: int
    identity: Identity
    producer: str
    component_id: int
    output_port: str
    parents: tuple[int, ...]
    payload_kind: str
    tensors: dict[str, int]
    meta: dict
    consumers_remaining: int = 1

    def record(self, store: "Runtime") -> dict:
        return {
            "token_id": self.token_id,
            **self.identity.__dict__,
            "producer_instance_id": self.producer,
            "component_id": self.component_id,
            "output_port": self.output_port,
            "parent_token_ids": list(self.parents),
            "payload_kind": self.payload_kind,
            "tensor_handles": self.tensors,
            "tensor_meta": {k: store.tensor_meta(h) for k, h in self.tensors.items()},
            **self.meta,
        }


@dataclass
class Pending:
    component_id: int
    identity: Identity
    parents: tuple[int, ...]
    payload_kind: str
    tensors: dict[str, torch.Tensor]
    meta: dict
    committed: bool = False


@dataclass
class KVState:
    k: torch.Tensor | None = None
    v: torch.Tensor | None = None
    valid_length: int = 0
    version: int = 0


def component_name(cid: int) -> tuple[int | None, str]:
    layer, op = divmod(cid, 16)
    if layer < TOP_BASE:
        if op not in OP_NAMES:
            raise BridgeError(f"component {cid}: layer {layer} has no op {op}")
        return layer, OP_NAMES[op]
    if layer == TOP_BASE and op in TOP_NAMES:
        return None, TOP_NAMES[op]
    raise BridgeError(f"unknown component id {cid}")


def producer_id(cid: int) -> str:
    layer, op = component_name(cid)
    return op if layer is None else f"layer.{layer:02d}.{op}"


@dataclass
class Runtime:
    run_id: str
    run_dir: Path
    case: str
    prompt: str | None = None
    max_decode_steps: int = 0
    eos_ids: tuple[int, ...] = ()
    model: object = None
    tokenizer: object = None
    reference: dict = field(default_factory=dict)
    artifacts: list[Artifact | None] = field(default_factory=lambda: [None])
    tensors: list[torch.Tensor | None] = field(default_factory=lambda: [None])
    pending: list[Pending | None] = field(default_factory=lambda: [None])
    kv: dict[tuple, KVState] = field(default_factory=dict)
    journal: list[dict] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    seq: int = 0

    # -- journal and failure -------------------------------------------------

    def log(self, event: str, **fields) -> None:
        self.seq += 1
        self.journal.append({"seq": self.seq, "event": event, "host_wall_time": time.time(), **fields})

    def fail(self, message: str) -> int:
        self.failures.append(message)
        self.log("failure", message=message)
        print(f"qwen bridge: FAILURE: {message}", file=sys.stderr, flush=True)
        return INVALID

    # -- tables ----------------------------------------------------------------

    def put_tensor(self, t: torch.Tensor) -> int:
        self.tensors.append(t)
        return len(self.tensors) - 1

    def tensor(self, handle: int) -> torch.Tensor:
        t = self.tensors[handle] if 0 < handle < len(self.tensors) else None
        if t is None:
            raise BridgeError(f"missing tensor handle {handle}")
        return t

    def tensor_meta(self, handle: int) -> dict:
        t = self.tensor(handle)
        return {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""),
                "bytes": t.numel() * t.element_size(), "device": str(t.device)}

    def artifact(self, token_id: int) -> Artifact:
        a = self.artifacts[token_id] if 0 < token_id < len(self.artifacts) else None
        if a is None:
            raise BridgeError(f"missing token id {token_id}")
        return a

    def publish(self, identity: Identity, producer: str, cid: int, port: str, parents: tuple,
                kind: str, tensors: dict[str, torch.Tensor], meta: dict | None = None) -> int:
        token_id = len(self.artifacts)
        if token_id > MAX_TOKEN:
            raise BridgeError(f"token id {token_id} overflows {STAGES['token_id_width']} bits")
        handles = {k: self.put_tensor(t) for k, t in tensors.items()}
        art = Artifact(token_id, identity, producer, cid, port, parents, kind, handles, meta or {})
        self.artifacts.append(art)
        self.log("commit", token_id=token_id, producer=producer, port=port, parents=list(parents),
                 identity=identity.__dict__, bytes=sum(self.tensor_meta(h)["bytes"] for h in handles.values()))
        return token_id

    def consume(self, token_id: int, consumer: str) -> Artifact:
        a = self.artifact(token_id)
        if a.consumers_remaining <= 0:
            raise BridgeError(f"{consumer}: token {token_id} from {a.producer} has no consumers left "
                              "(repeated delivery?)")
        a.consumers_remaining -= 1
        self.log("receive", token_id=token_id, consumer=consumer, remaining=a.consumers_remaining)
        return a

    # -- sources ------------------------------------------------------------

    def ensure_model(self) -> None:
        if self.model is None:
            torch.set_grad_enabled(False)
            self.model, self.tokenizer = load_model()
            self.log("model_loaded")

    def ensure_reference(self) -> None:
        if self.prompt is not None:
            return
        if not self.reference:
            path = ROOT / "runs" / "reference" / self.case / "capture.pt"
            if not path.exists():
                raise BridgeError(f"no reference capture at {path}; run reference/capture.py {self.case}")
            self.reference = torch.load(path, weights_only=False)

    def prompt_ids(self) -> torch.Tensor:
        if self.prompt is not None:
            self.ensure_model()
            ids = self.tokenizer(self.prompt, return_tensors="pt", add_special_tokens=False).input_ids
            if ids.shape[1] == 0:
                raise BridgeError("custom prompt tokenizes to nothing")
            return ids
        self.ensure_reference()
        return self.reference["input_ids"].clone()

    def prefill_identity(self, request_id: int) -> Identity:
        return Identity(self.run_id, request_id, 0, "prefill", 0, int(self.prompt_ids().shape[1]))

    def source_prompt(self, request_id: int) -> int:
        ident = self.prefill_identity(request_id)
        return self.publish(ident, "source", -1, "input_ids", (), "input_ids", {"input_ids": self.prompt_ids()})

    def continue_after(self, token_id: int) -> int:
        """Controller decision after a selected token: 1 continue, 2 end the request."""
        a = self.artifact(token_id)
        if a.payload_kind != "next_token":
            raise BridgeError(f"controller expected a selected token, got {a.payload_kind} from {a.producer}")
        tok = int(self._t(a, "token_id").item())
        if a.identity.invocation_id >= self.max_decode_steps:
            reason = "max_decode_steps"
        elif tok in self.eos_ids:
            reason = "eos"
        else:
            reason = None
        self.log("controller", token_id=token_id, vocab_id=tok, invocation_id=a.identity.invocation_id,
                 decision="end" if reason else "continue", reason=reason)
        return 2 if reason else 1

    def decode_input(self, token_id: int) -> int:
        """Next decode invocation: the selected token at the next position, sequence length 1."""
        a = self.consume(token_id, "controller")
        ident = a.identity
        nxt = Identity(ident.run_id, ident.request_id, ident.invocation_id + 1, "decode",
                       ident.position_start + ident.sequence_length, 1)
        tok = self._t(a, "token_id").reshape(1, 1).clone()
        return self.publish(nxt, "controller", -1, "input_ids", (token_id,), "input_ids", {"input_ids": tok})

    def source_layer_input(self, layer: int) -> int:
        """Reference input hidden state of one layer, for one-layer validation."""
        ident = self.prefill_identity(1)
        t = self.reference["tensors"][f"layers.{layer}.input"].clone()
        return self.publish(ident, f"reference.layer.{layer:02d}.input", -1, "hidden", (), "hidden", {"hidden": t})

    def fork(self, token_id: int, n: int) -> int:
        a = self.artifact(token_id)
        if n < 2:
            raise BridgeError(f"fork of token {token_id} needs at least two consumers, got {n}")
        if a.consumers_remaining != 1:
            raise BridgeError(f"fork of token {token_id}: expected one unclaimed consumer, "
                              f"found {a.consumers_remaining}")
        a.consumers_remaining = n
        self.log("fork", token_id=token_id, consumers=n)
        return 1

    # -- compute and commit ------------------------------------------------------

    def compute(self, cid: int, inputs: tuple[int, ...]) -> int:
        self.ensure_model()
        layer, op = component_name(cid)
        name = producer_id(cid)
        arts = [self.consume(t, name) for t in inputs]
        idents = {a.identity.join_key() for a in arts}
        if len(idents) != 1:
            raise BridgeError(f"{name}: join identity mismatch: " +
                              "; ".join(f"token {a.token_id} {a.identity.join_key()}" for a in arts))
        ident = arts[0].identity
        self.log("compute_start", component=name, inputs=list(inputs))
        kind, tensors, meta = self._dispatch(layer, op, arts)
        self.pending.append(Pending(cid, ident, tuple(inputs), kind, tensors, meta))
        self.log("compute_done", component=name, handle=len(self.pending) - 1)
        return len(self.pending) - 1

    def commit(self, cid: int, handle: int) -> int:
        p = self.pending[handle] if 0 < handle < len(self.pending) else None
        if p is None:
            raise BridgeError(f"missing pending handle {handle}")
        if p.committed:
            raise BridgeError(f"pending handle {handle} committed twice")
        if p.component_id != cid:
            raise BridgeError(f"pending handle {handle} belongs to component {p.component_id}, "
                              f"commit requested by {cid}")
        layer, op = component_name(cid)
        meta = dict(p.meta)
        if op == "cache_update":
            meta.update(self._apply_cache(layer, p))
        p.committed = True
        token = self.publish(p.identity, producer_id(cid), cid, op, p.parents, p.payload_kind, p.tensors, meta)
        self.pending[handle] = None
        return token

    # -- components --------------------------------------------------------------

    def _layer(self, layer: int):
        return self.model.model.layers[layer]

    def _t(self, art: Artifact, key: str) -> torch.Tensor:
        if key not in art.tensors:
            raise BridgeError(f"token {art.token_id} ({art.payload_kind} from {art.producer}) has no tensor {key}")
        return self.tensor(art.tensors[key])

    def _hidden(self, art: Artifact) -> torch.Tensor:
        h = self._t(art, "hidden")
        want = (1, art.identity.sequence_length, self.model.config.hidden_size)
        if tuple(h.shape) != want:
            raise BridgeError(f"token {art.token_id}: hidden shape {tuple(h.shape)}, expected {want}")
        return h

    def _positions(self, ident: Identity) -> torch.Tensor:
        return torch.arange(ident.position_start, ident.position_start + ident.sequence_length).unsqueeze(0)

    def _dispatch(self, layer, op, arts):
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, eager_attention_forward

        m = self.model
        a = arts[0]
        if layer is None:
            if op == "embed":
                return "hidden", {"hidden": m.model.embed_tokens(self._t(a, "input_ids"))}, {}
            if op == "final_norm":
                return "hidden", {"hidden": m.model.norm(self._hidden(a))}, {}
            if op == "lm_head":
                # Only the last position is needed to choose the next token.
                return "logits", {"logits": m.lm_head(self._hidden(a)[:, -1:, :])[:, 0, :]}, {}
            if op == "select":
                logits = self._t(a, "logits")
                if tuple(logits.shape) != (1, m.config.vocab_size):
                    raise BridgeError(f"token {a.token_id}: logits shape {tuple(logits.shape)}")
                return "next_token", {"token_id": logits.argmax(dim=-1)}, {}
            raise BridgeError(f"top-level op {op} is not implemented")

        L = self._layer(layer)
        attn = L.self_attn
        if op == "input_norm":
            return "hidden", {"hidden": L.input_layernorm(self._hidden(a))}, {}
        if op == "post_attention_norm":
            return "hidden", {"hidden": L.post_attention_layernorm(self._hidden(a))}, {}
        if op == "qkv_projection":
            x = self._hidden(a)
            return "qkv_bundle", {"q": attn.q_proj(x), "k": attn.k_proj(x), "v": attn.v_proj(x)}, {}
        if op == "qk_norm_rope":
            q, k, v = self._t(a, "q"), self._t(a, "k"), self._t(a, "v")
            shape = (1, a.identity.sequence_length, -1, attn.head_dim)
            q = attn.q_norm(q.view(shape)).transpose(1, 2)
            k = attn.k_norm(k.view(shape)).transpose(1, 2)
            v = v.view(shape).transpose(1, 2)
            cos, sin = m.model.rotary_emb(v, self._positions(a.identity))
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            return "qkv_heads", {"q": q, "k": k, "v": v}, {}
        if op == "cache_update":
            # Private copy until commit; the KV state is untouched here.
            return "q_and_cache", {"q": self._t(a, "q"), "k_new": self._t(a, "k"), "v_new": self._t(a, "v")}, {}
        if op == "attention":
            key = (a.identity.run_id, a.identity.request_id, layer)
            state = self.kv.get(key)
            if state is None:
                raise BridgeError(f"layer {layer}: no KV state for {key}")
            if state.version != a.meta.get("cache_version"):
                raise BridgeError(f"layer {layer}: stale cache version {a.meta.get('cache_version')}, "
                                  f"current {state.version}")
            end = a.identity.position_start + a.identity.sequence_length
            if state.valid_length != end:
                raise BridgeError(f"layer {layer}: cache holds {state.valid_length} positions, query ends at {end}")
            q = self._t(a, "q")
            s, t = q.shape[2], state.valid_length
            mask = torch.zeros(1, 1, s, t, dtype=q.dtype)
            qpos = torch.arange(a.identity.position_start, end).unsqueeze(1)
            mask.masked_fill_(torch.arange(t).unsqueeze(0) > qpos, torch.finfo(q.dtype).min)
            out, _ = eager_attention_forward(attn, q, state.k, state.v, mask, scaling=attn.scaling)
            return "attn_heads", {"hidden_heads": out.reshape(1, s, -1).contiguous()}, {"cache_version": state.version}
        if op == "attention_output_projection":
            return "hidden", {"hidden": attn.o_proj(self._t(a, "hidden_heads"))}, {}
        if op in ("attention_residual_add", "mlp_residual_add"):
            if len(arts) != 2:
                raise BridgeError(f"layer {layer} {op} needs two inputs, got {len(arts)}")
            return "hidden", {"hidden": self._hidden(arts[0]) + self._hidden(arts[1])}, {}
        if op == "mlp":
            return "hidden", {"hidden": L.mlp(self._hidden(a))}, {}
        raise BridgeError(f"layer op {op} is not implemented")

    def _apply_cache(self, layer: int, p: Pending) -> dict:
        ident = p.identity
        key = (ident.run_id, ident.request_id, layer)
        state = self.kv.setdefault(key, KVState())
        if ident.position_start != state.valid_length:
            raise BridgeError(f"layer {layer}: append at position {ident.position_start}, but cache holds "
                              f"{state.valid_length} positions (duplicate or missing append)")
        k_new, v_new = p.tensors.pop("k_new"), p.tensors.pop("v_new")
        state.k = k_new if state.k is None else torch.cat([state.k, k_new], dim=2)
        state.v = v_new if state.v is None else torch.cat([state.v, v_new], dim=2)
        state.valid_length += ident.sequence_length
        state.version += 1
        # torch.cat made new tensors, so this version's snapshot stays immutable.
        p.tensors["cache_k"], p.tensors["cache_v"] = state.k, state.v
        self.log("cache_append", layer=layer, request_id=ident.request_id, positions=[ident.position_start,
                 ident.position_start + ident.sequence_length], valid_length=state.valid_length, version=state.version)
        return {"cache_version": state.version, "cache_valid_length": state.valid_length}

    # -- validation ---------------------------------------------------------------

    def compare(self) -> dict:
        """Compare every committed artifact that has a reference counterpart.

        Prefill (invocation 0) uses unprefixed reference keys; decode invocation
        k uses keys prefixed "decode<k>.". A custom prompt has no reference.
        """
        if self.prompt is not None:
            return {"rows": []}
        self.ensure_reference()
        ref = self.reference["tensors"]
        rows = []
        for a in self.artifacts[1:]:
            if a is None or a.component_id < 0:
                continue
            layer, op = component_name(a.component_id)
            inv = a.identity.invocation_id
            prefix = "" if inv == 0 else f"decode{inv}."
            for name, ref_key, got in self._reference_pairs(layer, op, a):
                ref_key = prefix + ref_key
                if ref_key not in ref:
                    continue
                want = ref[ref_key]
                base = {"producer": a.producer, "invocation_id": inv, "tensor": name, "reference": ref_key}
                if tuple(got.shape) != tuple(want.shape):
                    rows.append({**base, "ok": False, "error": f"shape {tuple(got.shape)} vs {tuple(want.shape)}"})
                    continue
                diff = (got.float() - want.float()).abs()
                atol, rtol = TOLERANCE["logits" if op == "lm_head" else "stage"]
                ratio = float((diff / (atol + rtol * want.float().abs())).max())
                rel = diff / want.float().abs().clamp_min(atol)
                rows.append({**base, "dtype": str(got.dtype).replace("torch.", ""), "max_abs": float(diff.max()),
                             "max_rel": float(rel.max()), "atol": atol, "rtol": rtol,
                             "tolerance_ratio": ratio, "ok": ratio <= 1.0})
        return {"rows": rows}

    def _reference_pairs(self, layer, op, a: Artifact):
        if layer is None:
            if op == "embed":
                yield "hidden", "embed", self._t(a, "hidden")
            elif op == "final_norm":
                yield "hidden", "final_norm", self._t(a, "hidden")
            elif op == "lm_head":
                yield "logits", "logits_last", self._t(a, "logits")
                # Decode only: the reference recomputed the whole prefix without a cache.
                yield "logits_vs_full_prefix", "logits_full_prefix", self._t(a, "logits")
            return
        p = f"layers.{layer}."
        simple = {"input_norm": "input_norm", "attention_output_projection": "attn_out",
                  "attention_residual_add": "resid1", "post_attention_norm": "post_norm",
                  "mlp": "mlp_out", "mlp_residual_add": "layer_out"}
        if op in simple:
            yield "hidden", p + simple[op], self._t(a, "hidden")
        elif op == "qkv_projection":
            for k in ("q", "k", "v"):
                yield k, p + f"{k}_proj", self._t(a, k)
        elif op == "qk_norm_rope":
            yield "q", p + "q_rope", self._t(a, "q")
            if a.identity.phase == "prefill":
                # During decode the reference only exposes the whole cache, which cache_update covers.
                yield "k", p + "k_cache", self._t(a, "k")
                yield "v", p + "v_cache", self._t(a, "v")
        elif op == "cache_update":
            yield "cache_k", p + "k_cache", self._t(a, "cache_k")
            yield "cache_v", p + "v_cache", self._t(a, "cache_v")
        elif op == "attention":
            yield "hidden_heads", p + "attn_core", self._t(a, "hidden_heads")

    def _common_report(self, final_token: int, mode: str) -> tuple[dict, bool]:
        report = self.compare()
        leftovers = [{"token_id": x.token_id, "producer": x.producer, "remaining": x.consumers_remaining}
                     for x in self.artifacts[1:] if x is not None and x.token_id != final_token
                     and x.consumers_remaining != 0]
        unfinished = [i for i, p in enumerate(self.pending) if p is not None]
        report.update({
            "run_id": self.run_id, "case": self.case, "mode": mode,
            "final_token": self.artifact(final_token).record(self),
            "artifacts": len(self.artifacts) - 1, "failures": self.failures,
            "unconsumed_artifacts": leftovers, "uncommitted_handles": unfinished,
            "retained_tensor_bytes": sum(t.numel() * t.element_size() for t in self.tensors[1:] if t is not None),
            "cache": {f"{k[0]}/req{k[1]}/layer{k[2]}": {"valid_length": s.valid_length, "version": s.version}
                      for k, s in self.kv.items()},
        })
        rows = report["rows"]
        ok = not self.failures and all(r["ok"] for r in rows) and not leftovers and not unfinished
        return report, ok

    def _print_worst(self, rows: list[dict]) -> None:
        worst = max(rows, key=lambda r: r.get("tolerance_ratio", float("inf")), default=None)
        if worst:
            print(f"qwen bridge: {len(rows)} comparisons, worst {worst['producer']}.{worst['tensor']} "
                  f"(invocation {worst['invocation_id']}) max_abs {worst.get('max_abs')} "
                  f"tolerance ratio {worst.get('tolerance_ratio')}", file=sys.stderr, flush=True)
        else:
            print("qwen bridge: nothing compared", file=sys.stderr, flush=True)

    def finish_layer(self, token_id: int) -> int:
        report, ok = self._common_report(token_id, "layer")
        ok = ok and bool(report["rows"])
        report["ok"] = ok
        self.write(report)
        self._print_worst(report["rows"])
        return 1 if ok else 2

    def finish_generate(self, token_id: int) -> int:
        """End of request: the controller hands over the last selected token."""
        final = self.artifact(token_id)
        report, ok = self._common_report(token_id, "generate")
        selects = sorted((a for a in self.artifacts[1:] if a is not None and a.payload_kind == "next_token"),
                         key=lambda a: a.identity.invocation_id)
        ids = [int(self._t(a, "token_id").item()) for a in selects]
        prompt = self.prompt_ids()[0].tolist()
        gen = {"prompt": self.prompt if self.prompt is not None else self.reference["prompt"],
               "prompt_ids": prompt, "generated_ids": ids, "text": self.tokenizer.decode(ids),
               "invocations": [{"invocation_id": a.identity.invocation_id, "phase": a.identity.phase,
                                "position_start": a.identity.position_start,
                                "sequence_length": a.identity.sequence_length,
                                "selected_id": i} for a, i in zip(selects, ids)]}

        # Every layer's cache holds each position exactly once: prompt plus one per decode invocation.
        expect_len = final.identity.position_start + final.identity.sequence_length
        expect_version = final.identity.invocation_id + 1
        bad_cache = {k: v for k, v in report["cache"].items()
                     if (v["valid_length"], v["version"]) != (expect_len, expect_version)}
        if len(report["cache"]) != self.model.config.num_hidden_layers or bad_cache:
            ok = False
        gen["cache_expected"] = {"valid_length": expect_len, "version": expect_version,
                                 "layers": len(report["cache"]), "mismatched": bad_cache}

        if self.prompt is None:
            ref_ids = [int(self.reference["first_generated_id"])] + list(self.reference["generated_ids"])
            n = len(ids)
            gen["reference_ids"] = ref_ids[:n]
            gen["reference_text"] = self.tokenizer.decode(ref_ids[:n])
            gen["agree"] = ids == ref_ids[:n]
            ok = ok and gen["agree"] and bool(report["rows"])
        else:
            gen["agree"] = None
            gen["note"] = "custom prompt: no reference capture, numerics not validated"
        report["generation"] = gen
        report["ok"] = bool(ok)
        self.write(report)
        self._print_worst(report["rows"])
        print(f"qwen bridge: generated {ids} {gen['text']!r}", file=sys.stderr, flush=True)
        return 1 if ok else 2

    def write(self, report: dict) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with open(self.run_dir / "journal.jsonl", "w") as f:
            for e in self.journal:
                f.write(json.dumps(e) + "\n")
        with open(self.run_dir / "artifacts.jsonl", "w") as f:
            for a in self.artifacts[1:]:
                if a is not None:
                    f.write(json.dumps(a.record(self)) + "\n")
        with open(self.run_dir / "report.json", "w") as f:
            json.dump(report, f, indent=2)


# (atol, rtol), applied per element: |act - ref| <= atol + rtol * |ref|.
# A per-tensor bound scaled by max |ref| is not used: Qwen3 hidden states have
# outlier channels near 6.5e3 while the median is ~0.1-2, so it would hide
# errors in ordinary channels.
# Stages run the same FP32 CPU kernels on the same inputs as the reference, so
# the expected difference is zero or a few ulps. Logits get 1e-4: the LM head
# here multiplies only the last position (a different GEMM shape), and the
# reference's own cached and full-prefix logits differ by up to ~5e-5.
TOLERANCE = {"stage": (1e-5, 1e-5), "logits": (1e-4, 1e-4)}

RUNTIME: Runtime | None = None


def runtime() -> Runtime:
    global RUNTIME
    if RUNTIME is None:
        prompt = os.environ.get("QWEN_PROMPT") or None
        case = "custom" if prompt is not None else os.environ.get("QWEN_CASE", "capital")
        cases = {c["name"]: c for c in load_json("prompts.json")["cases"]}
        default_steps = cases[case]["decode_steps"] if case in cases else 0
        steps = int(os.environ.get("QWEN_DECODE_STEPS", default_steps))
        if prompt is None and case in cases and steps > default_steps:
            raise BridgeError(f"case {case} has a reference for {default_steps} decode steps, {steps} requested")
        eos = load_json("model.json")["eos_token_ids"]
        run_id = os.environ.get("QWEN_RUN_ID", time.strftime("run-%Y%m%d-%H%M%S"))
        run_dir = Path(os.environ.get("QWEN_RUN_DIR", ROOT / "runs" / run_id))
        RUNTIME = Runtime(run_id=run_id, run_dir=run_dir, case=case, prompt=prompt,
                          max_decode_steps=steps, eos_ids=tuple(eos))
        RUNTIME.log("run_start", case=case, prompt=prompt, max_decode_steps=steps, eos_token_ids=eos)
    return RUNTIME


def guarded(fn):
    """Exported entry points return 0 on any error, after recording it with context."""
    def wrapper(*args):
        rt = runtime()
        try:
            return fn(rt, *[int(x) for x in args])
        except BridgeError as exc:
            return rt.fail(f"{fn.__name__}{tuple(int(x) for x in args)}: {exc}")
        except Exception as exc:  # noqa: BLE001 - every failure must reach the report
            return rt.fail(f"{fn.__name__}{tuple(int(x) for x in args)}: {type(exc).__name__}: {exc}")
    wrapper.__name__ = fn.__name__
    return wrapper


@guarded
def init(rt: Runtime) -> int:
    rt.ensure_reference()
    rt.ensure_model()
    return 1


@guarded
def source_prompt(rt: Runtime, request_id: int) -> int:
    return rt.source_prompt(request_id)


@guarded
def source_layer(rt: Runtime, layer: int) -> int:
    return rt.source_layer_input(layer)


@guarded
def fork(rt: Runtime, token_id: int, n: int) -> int:
    return rt.fork(token_id, n)


@guarded
def compute1(rt: Runtime, cid: int, a: int) -> int:
    return rt.compute(cid, (a,))


@guarded
def compute2(rt: Runtime, cid: int, a: int, b: int) -> int:
    return rt.compute(cid, (a, b))


@guarded
def commit(rt: Runtime, cid: int, handle: int) -> int:
    return rt.commit(cid, handle)


@guarded
def continue_after(rt: Runtime, token_id: int) -> int:
    return rt.continue_after(token_id)


@guarded
def decode_input(rt: Runtime, token_id: int) -> int:
    return rt.decode_input(token_id)


@guarded
def finish_layer(rt: Runtime, token_id: int) -> int:
    return rt.finish_layer(token_id)


@guarded
def finish_generate(rt: Runtime, token_id: int) -> int:
    return rt.finish_generate(token_id)
