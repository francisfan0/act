# ACT model of Qwen3-0.6B

A functional transformer simulation in ACT. Each ACT process is one core
operation. Channels carry 32-bit artifact token ids. actsim loads a C bridge
that embeds CPython; the Python runtime computes with the pinned model's own
modules and records what each token id points at.

The model is decoder-only (`Qwen3ForCausalLM`): an embedding, 28 identical
decoder blocks with causal grouped-query self-attention, a final RMSNorm, and
a tied LM head. There is no encoder and no cross-attention.

## Setup

The bridge build expects macOS with Homebrew Python 3.12 (`python3.12-config`
at `/opt/homebrew/opt/python@3.12/bin/python3.12-config`). The first check
compiles `build/libqwen_bridge.dylib`.

```sh
scripts/install_toolchain.sh
```

This clones `act`, `stdlib`, `annotate`, and `actsim` into `$HOME/async/src`
and installs them under `$HOME/async`. The revisions this tree was run with
are in `config/toolchain.json`. Check those commits out and run the script
again to build them.

```sh
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv
.venv/bin/pip install torch==2.10.0 transformers==5.0.0 safetensors==0.7.0 numpy==1.26.4
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B', revision='c1899de289a04d12100db370d81485cdf75e47ca')"
```

Runs load that revision from the local Hugging Face cache only. They cast the
bfloat16 checkpoint to FP32 on CPU and use eager attention. Prompts are raw
text without the chat template, and selection is greedy.

## Layout

```text
graph/       ACT processes (qwen_lib.act) and the actsim extern config
runtime/     C ABI (bridge/bridge.cc) and the functional backend
reference/   pinned reference runner (capture.py)
config/      checkpoint, prompts, stage ids, memory, timing, toolchain revisions
scripts/     toolchain install, bridge build, graph generator, checks
tests/       functional-backend fault injection
runs/        generated captures, reports, traces (gitignored)
```

The checkpoint stores `lm_head.weight` and `model.embed_tokens.weight`
separately. They are bitwise equal, but transformers 5.0 loads them as two
tensors. `runtime/model.py` checks that they are equal, along with every layer
weight shape (Q 2048, K/V 1024, head dim 128).

## Checks

```sh
.venv/bin/python reference/capture.py    # reference tensors for every prompt case
.venv/bin/python tests/test_functional.py
scripts/validate_graph.sh layer 3        # one decomposed block, reference input
scripts/validate_graph.sh model counting # prefill + 4 cached decode steps, all checked
scripts/validate_graph.sh generate "Your prompt" 24   # any prompt; output not validated
```

`model` runs a reference case (`config/prompts.json`) and checks every stage of
every invocation. `generate` tokenizes any prompt, with no reference, and stops
after the given number of decode steps or at EOS. Its numerics are not
validated, but it still enforces the same identity, join, and cache rules.

`validate_graph.sh` writes `runs/<run_id>/` with the generated `graph.act`,
`actsim.log`, `report.json` (every tensor comparison), `journal.jsonl` (receive,
compute start and done, commit, fork, cache append, failures), and
`artifacts.jsonl` (the immutable artifact records with lineage and tensor
shapes).

## Results on this machine

Both prompt cases run prefill plus 4 cached decode steps through the full ACT
graph. Every stage of every layer in every invocation matches the reference,
and the generated text is identical.

| Case | Prompt | ACT output (= reference) |
| --- | --- | --- |
| `capital` | `The capital of France is` | ` Paris. The capital of` |
| `counting` | `One, two, three, four,` | ` five, six, seven` |

Every intermediate tensor and every KV cache snapshot is bitwise equal to the
reference, in prefill and in decode. The graph calls the same FP32 kernels on
the same shapes.

- **Prefill logits** differ by at most 2.6e-5. The graph's LM head multiplies
  only the last position; the reference multiplies all positions.
- **Decode logits** are bitwise equal to the reference's cached decode. They are
  within 5e-5 of a full-prefix recomputation without a cache.
- **KV cache** in every layer ends at length prompt + 4, version 5: each
  position was appended exactly once.

A custom 24-step generation (`generate`) takes about 4 seconds of host time.

A comparison passes when every element satisfies
\(|act - ref| \le atol + rtol \cdot |ref|\): `1e-5` for stages and `1e-4` for
logits. A bound scaled by the tensor's maximum would be too loose. Qwen3 hidden
states have outlier channels near 6.5e3, while the median is about 0.1 to 2. The
reference's own cached and full-prefix logits differ by up to 5e-5, which sets
the scale for the logits tolerance.

`tests/test_functional.py` shows the checks can fail. It confirms that a 1e-3
perturbation of one stage's output is reported at that stage. It also confirms
that a second append of the same cache positions, a token delivered beyond its
registered consumers, a join of different invocations, a stale cache version, a
missing token, and a commit by the wrong component are each rejected with
context.

**Decode loop.** `generation_controller` issues the prompt as invocation 0
(prefill, positions 0 to S-1). After `lm_head`, the `select` stage takes the
argmax. The controller receives that selected token and calls
`qwen::continue_after`:

- at the step limit or an EOS id (`151645`, `151643`), it calls
  `qwen::finish_generate` and stops;
- otherwise it calls `qwen::decode_input`, which publishes invocation k+1:
  phase decode, the next position, sequence length 1, and the selected token
  as input. It sends that token to the embedding.

The first decode therefore consumes the generated token, not the last prompt
token again. A request's next invocation cannot start until the previous
selection arrives, so one request's tokens do not overlap in the graph.

## Design

**Wire tokens.** actsim external functions take and return one integer each.
Structure and array arguments are rejected, so a token is a 32-bit id. `0` is
never valid. The id resolves to an immutable artifact record: run, request,
invocation, phase, position start, sequence length, producer instance
(`layer.03.qkv_projection`), output port, parent token ids, payload kind,
tensor handles, shapes, dtype, bytes.

**Component ids.** Each compute process is `stage1<CID>` or `stage2<CID>`, with
`CID = layer * 16 + op` from `config/stage_map.json`. The embedding, final norm,
and LM head use layer 28.

**Compute then commit.** A stage calls `qwen::compute` (computes into a private
pending result), then `qwen::commit` (publishes an immutable artifact and logs
it), and only then sends the token. Both return values are asserted.

**Input memory.** Every compute input port, including both join ports, has an
`input_memory`: one slot of token references, with blocking handoff. Dequeue
frees the slot; it does not move or copy the tensor, which stays in the store.

**Forks and joins.** `fork2` registers two consumers before sending to either.
Each compute counts down the consumers of the tokens it reads, and a token read
more times than registered is an error. Joins require equal
`(run, request, invocation, position start, sequence length)`.

**KV state.** It is separate from the artifact table and keyed by
`(run, request, layer)`. It changes only at the `cache_update` commit. A commit
fails unless it starts at the current valid length, so each position is
appended exactly once. Attention checks the cache version and valid length
carried by its query token.

**Components.** They call the loaded model's modules: `input_layernorm`,
`q_proj`/`k_proj`/`v_proj`, `q_norm`/`k_norm`, `rotary_emb` and
`apply_rotary_pos_emb`, `eager_attention_forward`, `o_proj`,
`post_attention_layernorm`, `mlp`, `norm`, and `lm_head`. The only attention
code written here is the causal mask.

**Reference capture.** It uses forward hooks, plus a wrapper around
transformers' module-level `eager_attention_forward`. The wrapper records
post-RoPE Q, the cache K/V attention actually read, and the pre-`o_proj`
output, while the pinned implementation still does the computation.

## Limitations

- The journal records host wall time. Simulation time is in `actsim.log`. A full
  prefill is 12,480 ticks at actsim's default 10 per statement; that is not a
  hardware time.
- All artifacts are kept for the whole run, so tensors are never reclaimed.
  Retention is about 10 to 16 MiB per 5 to 8 token prefill. Consumer counts are
  tracked and checked at the end of the run.
- `input_memory` is one slot per port. `config/memory.json` sets `byte_capacity`
  to null.
- A failed CHP `assert` is logged, and simulation continues. The bridge records
  every failure, and the sink's final check fails if any occurred.
- Batch size 1, one request, short unpadded prompts. No checked prompt stops on
  an EOS id inside its step limit.
