"""Minimal repro: coremltools 9 stateful fp16 model — convert+save OK, predict SIGBUSes.

Self-contained (pure torch + coremltools, random weights, no downloads). A small **stateful**
Core ML model with fp16 `keyCache`/`valueCache` States and a transformer-style attention block,
using **one 5-D state buffer sliced per layer** (`keyCache[i] = …`), SIGBUSes on predict()
("Failed to build the model execution plan … error code: -14" / ANECCompile FAILED) once the
KV-cache seq (or hidden width) crosses ~128, while a trivial state-increment control runs fine.

RESOLUTION (see docs/RESULTS.md → Task C-2 — the full stateful path is now validated on CPU):
HEARTH's working export avoids this by (1) using PER-LAYER separate state buffers instead of one
5-D state sliced per layer, (2) converting `compute_units=CPU_ONLY` (the `-14` is the **ANE
compiler**), and (3) `compute_precision=FLOAT32`. This file is kept as the minimal artifact for a
coremltools/Core-ML issue: the two things worth filing are (a) the hard SIGBUS from slicing a 5-D
fp16 state along its leading dim, and (b) the ANE compiler failing to plan a stateful fp16
attention graph above ~128 seq (reproduce by flipping the convert to `CPU_AND_NE`).

Bisection (random weights, 2 layers, stacked-5-D state as below): trivial control → OK; large VOCAB
alone → OK; large STATE_LEN or HIDDEN → SIGBUS; STATE_LEN 96 → OK, 128 → SIGBUS. So it scales with
stateful-attention tensor size, not weights — an ANE-tiling / MIL threshold, NOT an OS-build issue.

Observed on: Apple M3 Pro, macOS 26.4.2 (build 25E260), coremltools 9.0, torch 2.7.0 and 2.7.1
(both crash identically), Python 3.11. coremltools requires fp16 states ("State only support fp16
dtype").

Run:  python scripts/coreml_stateful_repro.py
Expect: "[control] predict OK", "[repro] convert+save OK", then a SIGBUS/-14 crash at
"[repro] predict …". Set STATE_LEN = 96 to see the same model predict fine.
"""

from __future__ import annotations

import platform

import numpy as np
import torch
import coremltools as ct

print(f"platform : {platform.platform()}")
print(f"coremltools {ct.__version__} | torch {torch.__version__} | np {np.__version__}")

# ---- minimal architecture (deliberately tiny EXCEPT STATE_LEN, to isolate the trigger) ----
LAYERS = 2
HIDDEN = 64
N_HEADS = 4
N_KV = 2               # grouped-query attention (n_heads % n_kv == 0)
HEAD_DIM = HIDDEN // N_HEADS
GROUPS = N_HEADS // N_KV
INTER = 128            # SwiGLU intermediate size
STATE_LEN = 128        # fixed KV-cache window — 128 CRASHES, 96 predicts fine (see docstring)
VOCAB = 128
NEG = -1e4             # fp16-safe mask fill


def rms_norm(x, w, eps=1e-6):
    v = x.float()
    v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    return (v.to(x.dtype)) * w


def repeat_kv(x, n):    # [b, n_kv, s, d] -> [b, n_kv*n, s, d]
    b, kv, s, d = x.shape
    if n == 1:
        return x
    return x[:, :, None, :, :].expand(b, kv, n, s, d).reshape(b, kv * n, s, d)


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(HIDDEN, N_HEADS * HEAD_DIM, bias=False)
        self.k = torch.nn.Linear(HIDDEN, N_KV * HEAD_DIM, bias=False)
        self.v = torch.nn.Linear(HIDDEN, N_KV * HEAD_DIM, bias=False)
        self.o = torch.nn.Linear(N_HEADS * HEAD_DIM, HIDDEN, bias=False)
        self.gate = torch.nn.Linear(HIDDEN, 2 * INTER, bias=False)
        self.down = torch.nn.Linear(INTER, HIDDEN, bias=False)
        self.n1 = torch.nn.Parameter(torch.ones(HIDDEN))
        self.n2 = torch.nn.Parameter(torch.ones(HIDDEN))
        self.scaling = HEAD_DIM ** -0.5


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, HIDDEN)
        self.layers = torch.nn.ModuleList(Layer() for _ in range(LAYERS))
        self.norm = torch.nn.Parameter(torch.ones(HIDDEN))
        self.head = torch.nn.Linear(HIDDEN, VOCAB, bias=False)
        shape = (LAYERS, 1, N_KV, STATE_LEN, HEAD_DIM)
        self.register_buffer("keyCache", torch.zeros(shape))
        self.register_buffer("valueCache", torch.zeros(shape))

    def forward(self, inputIds, causalMask, writePos):
        # inputIds [1,1]; causalMask [1,1,1,STATE_LEN]; writePos int32[1]
        oh = (torch.arange(STATE_LEN) == writePos).to(self.keyCache.dtype).view(1, 1, STATE_LEN, 1)
        h = self.embed(inputIds)
        for i, ly in enumerate(self.layers):
            r = h
            x = rms_norm(h, ly.n1)
            q = ly.q(x).view(1, 1, N_HEADS, HEAD_DIM).transpose(1, 2)
            k = ly.k(x).view(1, 1, N_KV, HEAD_DIM).transpose(1, 2)
            v = ly.v(x).view(1, 1, N_KV, HEAD_DIM).transpose(1, 2)
            # one-hot blend write of this token's KV at writePos (coremltools has no scatter-into-state)
            self.keyCache[i] = self.keyCache[i] * (1 - oh) + k * oh
            self.valueCache[i] = self.valueCache[i] * (1 - oh) + v * oh
            ka = repeat_kv(self.keyCache[i], GROUPS)
            va = repeat_kv(self.valueCache[i], GROUPS)
            attn = torch.matmul(q, ka.transpose(2, 3)) * ly.scaling + causalMask
            attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
            out = torch.matmul(attn, va).transpose(1, 2).contiguous().view(1, 1, N_HEADS * HEAD_DIM)
            h = r + ly.o(out)
            r = h
            x = rms_norm(h, ly.n2)
            g, up = ly.gate(x).chunk(2, dim=-1)
            h = r + ly.down(torch.nn.functional.silu(g) * up)
        return self.head(rms_norm(h, self.norm))


def convert_and_save(module, example, inputs, states, path):
    for p in module.parameters():
        p.requires_grad_(False)
    ep = torch.export.export(module.eval(), example).run_decompositions({})
    ml = ct.convert(
        ep,
        inputs=inputs,
        outputs=[ct.TensorType(name="logits", dtype=np.float16)],
        states=states,
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    ml.save(path)
    return path


def run_control():
    """A trivial stateful model — proves state read/write works on this machine."""
    class Counter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("acc", torch.zeros(1, 4))

        def forward(self, x):
            self.acc[:] = self.acc + x
            return self.acc * 1.0

    m = Counter().eval()
    for p in m.parameters():
        p.requires_grad_(False)
    ep = torch.export.export(m, (torch.zeros(1, 4),)).run_decompositions({})
    ml = ct.convert(
        ep,
        inputs=[ct.TensorType(name="x", shape=(1, 4), dtype=np.float16)],
        outputs=[ct.TensorType(name="logits", dtype=np.float16)],
        states=[ct.StateType(wrapped_type=ct.TensorType(shape=(1, 4), dtype=np.float16), name="acc")],
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    ml.save("/tmp/repro_control.mlpackage")
    r = ct.models.MLModel("/tmp/repro_control.mlpackage", compute_units=ct.ComputeUnit.CPU_ONLY)
    s = r.make_state()
    r.predict({"x": np.ones((1, 4), np.float16)}, state=s)
    y = r.predict({"x": np.ones((1, 4), np.float16)}, state=s)["logits"]
    print("[control] predict OK — state persists, y =", y.ravel().tolist())


def run_repro():
    m = Model().eval()
    cache_shape = (LAYERS, 1, N_KV, STATE_LEN, HEAD_DIM)
    example = (
        torch.zeros((1, 1), dtype=torch.long),
        torch.zeros((1, 1, 1, STATE_LEN)),
        torch.zeros(1, dtype=torch.long),
    )
    inputs = [
        ct.TensorType(name="inputIds", shape=(1, 1), dtype=np.int32),
        ct.TensorType(name="causalMask", shape=(1, 1, 1, STATE_LEN), dtype=np.float16),
        ct.TensorType(name="writePos", shape=(1,), dtype=np.int32),
    ]
    states = [
        ct.StateType(wrapped_type=ct.TensorType(shape=cache_shape, dtype=np.float16), name="keyCache"),
        ct.StateType(wrapped_type=ct.TensorType(shape=cache_shape, dtype=np.float16), name="valueCache"),
    ]
    path = convert_and_save(m, example, inputs, states, "/tmp/repro_stateful.mlpackage")
    print("[repro] convert+save OK ->", path)

    for cu_name, cu in [("CPU_ONLY", ct.ComputeUnit.CPU_ONLY), ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE)]:
        print(f"[repro] loading + predict on {cu_name} … (expected: -14 / SIGBUS)")
        r = ct.models.MLModel(path, compute_units=cu)
        s = r.make_state()
        mask = np.zeros((1, 1, 1, STATE_LEN), np.float16)
        y = r.predict({"inputIds": np.array([[1]], np.int32), "causalMask": mask,
                       "writePos": np.array([0], np.int32)}, state=s)["logits"]
        print(f"[repro] {cu_name} predict OK — logits shape {y.shape}  (did NOT reproduce)")


if __name__ == "__main__":
    run_control()
    run_repro()
