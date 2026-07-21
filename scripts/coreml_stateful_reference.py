"""Stateful KV-cache Core ML export — VALIDATED reference recipe (ADR-011, Approach B).

STATUS (2026-07-20, Apple M3 Pro / macOS 26.4 / coremltools 9.0 / torch 2.7.x): ✅ end-to-end.
`Qwen2.5-0.5B-Instruct` exports to a stateful `.mlpackage`, runs fully offline in the CoreML
runtime, and greedy-matches the stock PyTorch model **token-for-token**:
`"The capital of France is"` → `" Paris. It is the largest city in Europe and the third"`.

The three fixes that got it from "hard SIGBUS" to greedy parity (see docs/RESULTS.md → Task C-2):
  1. PER-LAYER separate state buffers (keyCache{i}/valueCache{i}), NOT one 5-D `keyCache[N,…]`
     sliced per layer. Slicing a 5-D state along the layer dim crashes CoreML's execution-plan
     builder (hard SIGBUS) above ~128 seq.
  2. Convert with `compute_units=CPU_ONLY`. The `-14` / "Failed to build the model execution plan"
     was the **ANE compiler** (`ANECCompile FAILED`); CPU-only conversion removes it entirely.
  3. `compute_precision=FLOAT32` (states stay fp16 — coremltools mandates fp16 states). fp16
     *compute* made the decode degenerate into repetition; fp32 compute gives exact parity.

Remaining follow-up (a real, separately-filable coremltools/ANE issue — NOT an OS-build problem):
the ANE compiler can't build a plan for this stateful fp16 graph above ~128 seq. Minimal repro:
`scripts/coreml_stateful_repro.py`. Until that's fixed, this path runs on CPU (fine for the 0.5B;
the 0.5B on CPU is fast). ANE acceleration is the optimization left on the table.

The recipe (folds into src/hearth/coreml.py):
  * Own the Qwen2 attention core (reuse the HF weight modules) so the KV write position is an
    explicit input — HF's 4.57 `Cache` is churny and torch.export bans both data-dependent slice
    bounds and module-attribute mutation.
  * FULLY STATIC shapes: inputIds [1,1], causalMask [1,1,1,STATE_LEN] (fixed), writePos int32[1].
    Read the FULL fixed cache; the mask gates valid positions.
  * Write the new token's KV at writePos via a one-hot BLEND (cache*(1-oh)+kv*oh) — coremltools'
    EXIR frontend supports no index_copy/scatter-into-state, but the elementwise blend converts.

NOTE — contract divergence: this static (writePos + full-width mask) contract does NOT match
swift-transformers' `LanguageModelWithStatefulKVCache` (ranged inputIds + a [1,1,1,tokenCount+1]
mask, no writePos). Driving this model needs a small custom Swift decode loop in
CoreMLGeneration.swift — so landing Approach B is not, after all, a "no Swift change" drop-in.

Run:  python scripts/coreml_stateful_reference.py            # parity + export + convert + predict
      python scripts/coreml_stateful_reference.py --no-predict # skip the CoreML predict/parity step
"""

from __future__ import annotations

import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
STATE_LEN = 256  # fixed KV-cache window (max context)
NEG = -1e4  # fp16-safe mask fill (float32.min overflows fp16 to -inf)
DO_PREDICT = "--no-predict" not in sys.argv  # CoreML predict + greedy parity (validated path)

cfg = AutoConfig.from_pretrained(MODEL)
N_LAYERS = cfg.num_hidden_layers
N_KV = cfg.num_key_value_heads
HEAD_DIM = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
print(f"layers={N_LAYERS} kv_heads={N_KV} head_dim={HEAD_DIM} vocab={cfg.vocab_size}")

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, attn_implementation="eager")
model.eval()
tok = AutoTokenizer.from_pretrained(MODEL)


class StatefulQwen(torch.nn.Module):
    """Single-token stateful Qwen2 forward reusing the HF weight modules; fully static shapes."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        # PER-LAYER separate state buffers (keyCache{i}/valueCache{i}) — NOT one stacked 5-D
        # `keyCache[N_LAYERS,…]` sliced per layer. Slicing a 5-D state along the layer dim is what
        # breaks CoreML's execution-plan builder (-14 / SIGBUS) above ~128 seq; separate per-layer
        # states with in-place writes convert & run. See docs/RESULTS.md → Task C-2.
        for i in range(N_LAYERS):
            self.register_buffer(f"keyCache{i}", torch.zeros(1, N_KV, STATE_LEN, HEAD_DIM))
            self.register_buffer(f"valueCache{i}", torch.zeros(1, N_KV, STATE_LEN, HEAD_DIM))

    def _attn(self, sa, hidden, cos, sin, causalMask, oh, layer_idx):
        input_shape = hidden.shape[:-1]
        hidden_shape = (*input_shape, -1, HEAD_DIM)
        q = sa.q_proj(hidden).view(hidden_shape).transpose(1, 2)
        k = sa.k_proj(hidden).view(hidden_shape).transpose(1, 2)
        v = sa.v_proj(hidden).view(hidden_shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        # One-hot blend write of the new token's KV at writePos into this layer's cache (in place).
        kc = getattr(self, f"keyCache{layer_idx}")
        vc = getattr(self, f"valueCache{layer_idx}")
        kc[:] = kc * (1 - oh) + k * oh
        vc[:] = vc * (1 - oh) + v * oh
        k_all = repeat_kv(kc, sa.num_key_value_groups)
        v_all = repeat_kv(vc, sa.num_key_value_groups)
        attn = torch.matmul(q, k_all.transpose(2, 3)) * sa.scaling + causalMask
        attn = torch.nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v_all).transpose(1, 2).contiguous().reshape(*input_shape, -1)
        return sa.o_proj(out)

    def forward(self, inputIds, causalMask, writePos):
        # inputIds [1,1]; causalMask [1,1,1,STATE_LEN] (0 attend / NEG mask); writePos int32[1].
        inner = self.model.model
        oh = (torch.arange(STATE_LEN) == writePos).to(self.keyCache0.dtype).view(1, 1, STATE_LEN, 1)
        position_ids = writePos.view(1, 1).long()
        hidden = inner.embed_tokens(inputIds)
        cos, sin = inner.rotary_emb(hidden, position_ids)
        for i, layer in enumerate(inner.layers):
            residual = hidden
            h = layer.input_layernorm(hidden)
            h = self._attn(layer.self_attn, h, cos, sin, causalMask, oh, i)
            hidden = residual + h
            residual = hidden
            h = layer.post_attention_layernorm(hidden)
            hidden = residual + layer.mlp(h)
        hidden = inner.norm(hidden)
        return self.model.lm_head(hidden)


wrapper = StatefulQwen(model).eval()
for _p in wrapper.parameters():  # aot_export bans mutating grad-requiring graph inputs
    _p.requires_grad_(False)


def _zero_cache():
    for i in range(N_LAYERS):
        getattr(wrapper, f"keyCache{i}").zero_()
        getattr(wrapper, f"valueCache{i}").zero_()


def mask_for(pos):
    """[1,1,1,STATE_LEN]: token at absolute `pos` attends cache slots 0..pos, masks the rest."""
    m = torch.full((1, 1, 1, STATE_LEN), NEG)
    m[0, 0, 0, : pos + 1] = 0.0
    return m


def greedy_stateful(prompt, n=12):
    _zero_cache()
    ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
    logits = None
    for pos, t in enumerate(ids):  # prefill one token at a time
        with torch.no_grad():
            logits = wrapper(torch.tensor([[t]]), mask_for(pos), torch.tensor([pos]))
    pos = len(ids) - 1
    cur = int(logits[0, -1].argmax()); out = [cur]
    for _ in range(n - 1):
        pos += 1
        with torch.no_grad():
            logits = wrapper(torch.tensor([[cur]]), mask_for(pos), torch.tensor([pos]))
        cur = int(logits[0, -1].argmax()); out.append(cur)
    return out


def greedy_plain(prompt, n=12):
    seq = tok(prompt, return_tensors="pt").input_ids
    out = []
    for _ in range(n):
        with torch.no_grad():
            logits = model(input_ids=seq).logits
        cur = int(logits[0, -1].argmax()); out.append(cur)
        seq = torch.cat([seq, torch.tensor([[cur]])], dim=1)
    return out


PROMPT = "The capital of France is"
s = greedy_stateful(PROMPT); p = greedy_plain(PROMPT)
print("stateful:", tok.decode(s))
print("plain   :", tok.decode(p))
print("PARITY", "OK" if s == p else f"MISMATCH {s} != {p}")
if s != p:
    raise SystemExit("fix parity before exporting")

# --- torch.export + coremltools convert with States ------------------------------------------
import coremltools as ct

_zero_cache()
ex = (torch.zeros((1, 1), dtype=torch.long), torch.zeros((1, 1, 1, STATE_LEN)), torch.zeros(1, dtype=torch.long))
print("exporting…")
exported = torch.export.export(wrapper, ex).run_decompositions({})
print("export OK; converting…")
per_layer = (1, N_KV, STATE_LEN, HEAD_DIM)
states = []
for i in range(N_LAYERS):
    states.append(ct.StateType(wrapped_type=ct.TensorType(shape=per_layer, dtype=np.float16), name=f"keyCache{i}"))
    states.append(ct.StateType(wrapped_type=ct.TensorType(shape=per_layer, dtype=np.float16), name=f"valueCache{i}"))
mlmodel = ct.convert(
    exported,
    inputs=[
        ct.TensorType(name="inputIds", shape=(1, 1), dtype=np.int32),
        ct.TensorType(name="causalMask", shape=(1, 1, 1, STATE_LEN), dtype=np.float16),
        ct.TensorType(name="writePos", shape=(1,), dtype=np.int32),
    ],
    outputs=[ct.TensorType(name="logits", dtype=np.float16)],
    states=states,
    minimum_deployment_target=ct.target.macOS15,
    compute_precision=ct.precision.FLOAT32,   # fp16 compute degenerates the decode; fp32 -> parity
    compute_units=ct.ComputeUnit.CPU_ONLY,     # the ANE compiler fails (-14); CPU-only avoids it
)
out = "/tmp/qwen05-stateful.mlpackage"
mlmodel.save(out)
print("convert+save OK ->", out)

if not DO_PREDICT:
    print("skipping CoreML predict (--no-predict)")
    raise SystemExit(0)

# --- CoreML predict + greedy parity (validated: matches PyTorch token-for-token) --------------
for cu_name, cu in [("CPU_ONLY", ct.ComputeUnit.CPU_ONLY)]:
    r = ct.models.MLModel(out, compute_units=cu)
    st = r.make_state()
    ids = tok(PROMPT, return_tensors="pt").input_ids[0].tolist()
    logits = None
    for pos, t in enumerate(ids):
        logits = r.predict({"inputIds": np.array([[t]], np.int32),
                            "causalMask": mask_for(pos).to(torch.float16).numpy(),
                            "writePos": np.array([pos], np.int32)}, state=st)["logits"]
    pos = len(ids) - 1
    cur = int(logits[0, -1].argmax()); c = [cur]
    for _ in range(11):
        pos += 1
        logits = r.predict({"inputIds": np.array([[cur]], np.int32),
                            "causalMask": mask_for(pos).to(torch.float16).numpy(),
                            "writePos": np.array([pos], np.int32)}, state=st)["logits"]
        cur = int(logits[0, -1].argmax()); c.append(cur)
    print(f"coreml[{cu_name}]:", tok.decode(c))
    print(f"COREML[{cu_name}] PARITY", "OK" if c == p else f"divergence {c} != {p}")
