"""Stateful KV-cache Core ML export — validated reference recipe (ADR-011, Approach B).

STATUS (2026-07-20, Apple M3 Pro / macOS 26.4 Internal / coremltools 9.0 / torch 2.7.1):
  ✅ Greedy PARITY  — the recipe below reproduces the stock PyTorch model token-for-token.
  ✅ torch.export   — the stateful graph exports and lowers (`run_decompositions`).
  ✅ ct.convert + save — a stateful `.mlpackage` with keyCache/valueCache States is produced.
  ⛔ CoreML predict — running the compiled model SIGBUSes with "Failed to build the model
     execution plan … error code -14" (ANE compile also fails). Minimal synthetic stateful
     models (counter / one-hot blend / per-layer slice write) predict fine on this same stack,
     so the crash is the real-transformer-ops + fp16-states combination in the CoreML runtime,
     not a logic error. Suspected environmental: macOS 26 is an *Internal* build and coremltools
     flags torch 2.7.1 as untested (2.7.0 is the ceiling). See docs/RESULTS.md → Task C-2.

The recipe (folds into src/hearth/coreml.py once the runtime cooperates):
  * Own the Qwen2 attention core (reuse the HF weight modules) so the KV write position is an
    explicit shape symint / input — HF's 4.57 `Cache` is churny and torch.export bans both
    data-dependent slice bounds and module-attribute mutation.
  * FULLY STATIC shapes: inputIds [1,1], causalMask [1,1,1,STATE_LEN] (fixed), writePos int32[1].
    Read the FULL fixed cache; the mask gates valid positions. A dynamic-length `:end` cache read
    is what first produced the -14 (the state execution plan needs static intermediates).
  * Write the new token's KV at writePos via a one-hot BLEND (cache*(1-oh)+kv*oh) — coremltools'
    EXIR frontend supports no index_copy/scatter-into-state, but the elementwise blend converts.
  * fp16 States named keyCache/valueCache (coremltools mandates fp16 states).

NOTE — contract divergence: this static (writePos + full-width mask) contract does NOT match
swift-transformers' `LanguageModelWithStatefulKVCache` (ranged inputIds + a [1,1,1,tokenCount+1]
mask, no writePos). Driving this model needs a small custom Swift decode loop in
CoreMLGeneration.swift — so landing Approach B is not, after all, a "no Swift change" drop-in.

Run:  python scripts/coreml_stateful_reference.py            # parity + export + convert
      python scripts/coreml_stateful_reference.py --predict  # + attempt CoreML predict (may crash)
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
TRY_PREDICT = "--predict" in sys.argv  # the CoreML predict step SIGBUSes on the current stack

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
        shape = (N_LAYERS, 1, N_KV, STATE_LEN, HEAD_DIM)
        self.register_buffer("keyCache", torch.zeros(shape))
        self.register_buffer("valueCache", torch.zeros(shape))

    def _attn(self, sa, hidden, cos, sin, causalMask, oh, layer_idx):
        input_shape = hidden.shape[:-1]
        hidden_shape = (*input_shape, -1, HEAD_DIM)
        q = sa.q_proj(hidden).view(hidden_shape).transpose(1, 2)
        k = sa.k_proj(hidden).view(hidden_shape).transpose(1, 2)
        v = sa.v_proj(hidden).view(hidden_shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        # One-hot blend write of the new token's KV at writePos into the fixed cache (in place).
        self.keyCache[layer_idx] = self.keyCache[layer_idx] * (1 - oh) + k * oh
        self.valueCache[layer_idx] = self.valueCache[layer_idx] * (1 - oh) + v * oh
        k_all = repeat_kv(self.keyCache[layer_idx], sa.num_key_value_groups)
        v_all = repeat_kv(self.valueCache[layer_idx], sa.num_key_value_groups)
        attn = torch.matmul(q, k_all.transpose(2, 3)) * sa.scaling + causalMask
        attn = torch.nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v_all).transpose(1, 2).contiguous().reshape(*input_shape, -1)
        return sa.o_proj(out)

    def forward(self, inputIds, causalMask, writePos):
        # inputIds [1,1]; causalMask [1,1,1,STATE_LEN] (0 attend / NEG mask); writePos int32[1].
        inner = self.model.model
        oh = (torch.arange(STATE_LEN) == writePos).to(self.keyCache.dtype).view(1, 1, STATE_LEN, 1)
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


def mask_for(pos):
    """[1,1,1,STATE_LEN]: token at absolute `pos` attends cache slots 0..pos, masks the rest."""
    m = torch.full((1, 1, 1, STATE_LEN), NEG)
    m[0, 0, 0, : pos + 1] = 0.0
    return m


def greedy_stateful(prompt, n=12):
    wrapper.keyCache.zero_(); wrapper.valueCache.zero_()
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

wrapper.keyCache.zero_(); wrapper.valueCache.zero_()
ex = (torch.zeros((1, 1), dtype=torch.long), torch.zeros((1, 1, 1, STATE_LEN)), torch.zeros(1, dtype=torch.long))
print("exporting…")
exported = torch.export.export(wrapper, ex).run_decompositions({})
print("export OK; converting…")
cache_shape = (N_LAYERS, 1, N_KV, STATE_LEN, HEAD_DIM)
mlmodel = ct.convert(
    exported,
    inputs=[
        ct.TensorType(name="inputIds", shape=(1, 1), dtype=np.int32),
        ct.TensorType(name="causalMask", shape=(1, 1, 1, STATE_LEN), dtype=np.float16),
        ct.TensorType(name="writePos", shape=(1,), dtype=np.int32),
    ],
    outputs=[ct.TensorType(name="logits", dtype=np.float16)],
    states=[
        ct.StateType(wrapped_type=ct.TensorType(shape=cache_shape, dtype=np.float16), name="keyCache"),
        ct.StateType(wrapped_type=ct.TensorType(shape=cache_shape, dtype=np.float16), name="valueCache"),
    ],
    minimum_deployment_target=ct.target.macOS15,
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
)
out = "/tmp/qwen05-stateful.mlpackage"
mlmodel.save(out)
print("convert+save OK ->", out)

if not TRY_PREDICT:
    print("skipping CoreML predict (SIGBUSes on the current stack; pass --predict to attempt)")
    raise SystemExit(0)

# --- Attempt CoreML predict (KNOWN BLOCKER: SIGBUS / -14 on the current stack) ----------------
for cu_name, cu in [("CPU_ONLY", ct.ComputeUnit.CPU_ONLY), ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE)]:
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
