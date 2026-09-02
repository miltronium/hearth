# HEARTH — Local Model Roster (MLX / Apple M3 Pro 36 GB)

**Researched:** 2026-09-02
**Target machine:** Apple M3 Pro, 36 GiB unified memory, 237 GB free disk, macOS 26.4.2
**Runtime:** `mlx` 0.32.0 (installed) + `mlx-lm` / `mlx-vlm`

> **Verification policy for this document.** Every repo id in the recommendation tables was
> resolved live against `https://huggingface.co/api/models/<id>?blobs=true` on 2026-09-02, and the
> "Disk" column is the **sum of actual blob sizes returned by the API**, not an estimate.
> Architecture numbers (layer counts, KV heads, head dim, context) come from each repo's live
> `config.json`. Anything I could not verify is explicitly marked **UNVERIFIED**.
> Benchmarks/quality claims are third-party and marked as such.

---

## 0. TL;DR — one pick per role

| # | Role | Pick | Repo id | Disk | Resident (weights) | Verified |
|---|------|------|---------|------|--------------------|----------|
| 1 | General workhorse | **Qwen3.5-9B 4-bit** | `mlx-community/Qwen3.5-9B-4bit` | 5.98 GB | ~6.0 GB | ✅ API |
| 2 | Fine-tune target | **Qwen3.5-2B bf16** (LoRA) | `mlx-community/Qwen3.5-2B-bf16` | 4.45 GB | ~4.4 GB | ✅ API |
| 3 | Embeddings (RAG) | **Qwen3-Embedding-0.6B** | `Qwen/Qwen3-Embedding-0.6B` | 1.21 GB | ~1.2 GB | ✅ API |
| 4 | Offline LLM judge | **Qwen3.8-27B 4-bit** | `mlx-community/Qwen3.8-27B-4bit` | 16.08 GB | ~16.1 GB | ✅ API |
| 5 | Vision / scanned PDF | **dots.ocr 4-bit** | `mlx-community/dots.ocr-4bit` | 3.54 GB | ~3.5 GB | ✅ API |

**Total disk for the full roster: ~31 GB** (of 237 GB free). Comfortable.

### ⚠️ Answer to your direct question

**`mlx-community/Qwen2.5-14B-Instruct-4bit` is the wrong call. Cancel it.**
Replace with `mlx-community/Qwen3.5-9B-4bit`, which is *smaller, faster, longer-context, and
much stronger*. Details in [§5](#5-the-qwen25-14b-question).

### ⚠️ Toolchain blocker

Your installed `mlx-lm` is **0.29.1**. Qwen3.5-class architectures (`qwen3_5`) landed in
**mlx-lm 0.30.7**. Nothing in the Qwen3.5/3.6/3.8 or Gemma-4 families will load until you upgrade.
See [§7](#7-toolchain-upgrade-required-blocking).

---

## 1. Measured machine facts

These are **measured on your Mac**, not looked up:

```
$ python3 -c "import mlx.core as mx; print(mx.device_info())"
device_name                      = Apple M3 Pro
architecture                     = applegpu_g15s
memory_size                      = 38,654,705,664 B  = 38.65 GB (36.0 GiB)
max_recommended_working_set_size = 30,150,672,384 B  = 30.15 GB   <-- GPU ceiling
max_buffer_length                = 22,613,000,192 B  = 22.61 GB   <-- single-array cap
$ sysctl iogpu.wired_limit_mb  ->  0  (auto)
```

| Quantity | Value | What it means |
|---|---|---|
| Total unified memory | 38.65 GB | 36 GiB |
| **GPU working-set ceiling** | **30.15 GB** | Exceed this and macOS starts compressing/swapping; tok/s falls off a cliff |
| Single-array cap | 22.61 GB | No single MLX tensor may exceed this. None of these picks come close |
| Memory bandwidth | **150 GB/s** | M3 Pro spec — this is what caps generation speed |
| Free disk | 237 GB | Not a constraint for this roster |

---

## 2. The memory math (rules of thumb you can actually use)

### 2.1 Weights per billion parameters

MLX affine quantization at `group_size: 64` stores, per weight, the quantized bits **plus** an
fp16 scale and fp16 bias per 64-weight group — i.e. `bits + 32/64` effective bits.

| Precision | Theoretical bits/param | **GB per 1B params** | Verified against |
|---|---|---|---|
| 4-bit (g64) | 4.5 | **0.56** | Qwen2.5-14B-4bit: 8.31 GB / 14.77 B = **0.563** ✅ |
| 4-bit, mixed (real mlx-community repos) | ~4.7 | **0.59** | Qwen3.8-27B-4bit: 16.05 / 27.36 = **0.587** ✅ ; gemma-4-31b-4bit: 18.41 / 31.27 = **0.589** ✅ |
| 6-bit (g64) | 6.5 | **0.81–0.87** | Qwen3.5-9B-6bit: 8.19 / 9.41 = **0.87** ✅ |
| 8-bit (g64) | 8.5 | **1.06–1.11** | Qwen3.8-27B-8bit: 29.5 / 27.36 = **1.078** ✅ |
| bf16 | 16 | **2.00** | Qwen3.5-4B-bf16: 9.08 / 4.54 = **2.00** ✅ |

> **Planning number: 4-bit ≈ 0.60 GB per 1B params.** (Repos that keep embeddings or MoE gates at
> 8-bit run higher; QAT repos much higher — `gemma-4-12B-it-qat-4bit` is 0.92 GB/B because its MLPs
> are 8-bit.)

**Weights are resident, not transient.** Measured: `Qwen2.5-Coder-14B-Instruct-4bit` reports
8.309 GB active immediately after `mlx_lm.load()`, exactly matching its 8.31 GB of safetensors.

### 2.2 KV cache growth — the formula, validated on this machine

```
KV bytes per token = 2 (K+V) × n_kv_heads × head_dim × N_unbounded_attention_layers × 2 (fp16)
```

`N_unbounded_attention_layers` is the important term and it is **not** `num_hidden_layers` for any
modern model:

* **Qwen3.5 / 3.6 / 3.8** (`qwen3_5`): hybrid gated-delta **linear attention**, `full_attention_interval: 4`
  → only **1 in 4** layers keeps a growing KV cache. The other 3 hold a *constant-size* recurrent state.
* **Gemma 4** (`gemma4`): 5 sliding-window layers (window 1024) per 1 full-attention layer
  → only the full layers grow; sliding layers cap at 1024 tokens each.
* **Qwen2.5 / Qwen3** (`qwen2` / `qwen3`): every layer grows. Worst case.

**Empirical validation — measured on this Mac** (`mlx-community/Qwen2.5-Coder-14B-Instruct-4bit`,
batch 1, weights resident 8.309 GB):

| Prompt tokens | Peak GB | KV + workspace GB | Prompt tok/s |
|---:|---:|---:|---:|
| 256 | 8.761 | 0.452 | 147 |
| 2,000 | 9.388 | 1.079 | 171 |
| 8,000 | 10.614 | 2.304 | 158 |
| 16,000 | 12.111 | 3.802 | 152 |
| 32,000 | 15.225 | 6.916 | 120 |

Linear fit over 2k→32k gives **194.6 KB/token**, with a **~0.69 GB fixed intercept**.
Formula prediction: `2 × 8 kv_heads × 128 head_dim × 48 layers × 2 B` = **192 KB/token**.
**Agreement within 2% across a 16× context range.** The formula is trustworthy — use it.

Note also that **prompt processing slows as context grows** (171 → 120 tok/s from 2k to 32k), because
attention is quadratic. Ingesting a 32k-token statement into this 14B took **267 seconds**. On
Qwen3.5-9B, with 3 of every 4 layers using linear attention, that penalty is far smaller — another
reason it is the better workhorse for long documents.

### 2.3 KV cost per model (computed from live configs)

| Model | Layers | Unbounded attn layers | kv_heads | head_dim | **KB / token** | Fixed extra |
|---|---:|---:|---:|---:|---:|---|
| `Qwen2.5-14B-Instruct-4bit` | 48 | 48 | 8 | 128 | **192** | — |
| `Qwen3-14B-4bit` | 40 | 40 | 8 | 128 | **160** | — |
| `gemma-4-31b-it-4bit` | 60 | 10 full | 16 | 256 | **160** | 0.82 GB (50 sliding × 1024) |
| `Qwen3.8-27B-4bit` | 64 | 16 full | 4 | 256 | **64** | 0.15 GB linear state |
| `gemma-4-12B-it-qat-4bit` | 48 | 8 full | 8 | 256 | **64** | 0.33 GB (40 sliding × 1024) |
| `Qwen3.5-9B-4bit` | 32 | 8 full | 4 | 256 | **32** | 0.05 GB linear state |
| `Qwen3.5-4B-4bit` | 32 | 8 full | 4 | 256 | **32** | 0.05 GB |
| `Qwen3.6-35B-A3B-4bit` | 40 | 10 full | 2 | 256 | **20** | 0.06 GB |
| `Qwen3.5-2B-4bit` | 24 | 6 full | 2 | 256 | **12** | 0.03 GB |
| `gpt-oss-20b-MXFP4-Q8` | 24 | ~12 full | 8 | 64 | **~24** | ~0.003 GB (window 128) |

**This is the single most important table for financial documents.** A 100-page statement dump at
60k tokens costs **11.5 GB of KV** on Qwen2.5-14B but only **1.9 GB** on Qwen3.5-9B — a 6× difference
that completely changes what fits.

### 2.4 Total inference footprint

```
Peak GB  =  weights  +  (context_tokens × KB_per_token / 1,048,576)  +  fixed_state  +  0.7 GB workspace
```

| Model | 8k ctx | 32k ctx | 64k ctx | 128k ctx |
|---|---:|---:|---:|---:|
| `Qwen3.5-9B-4bit` | **6.9 GB** | **7.8 GB** | **8.7 GB** | **10.9 GB** |
| `Qwen3.5-4B-MLX-4bit` | 4.0 GB | 4.8 GB | 5.8 GB | 7.9 GB |
| `Qwen3.8-27B-4bit` | **17.4 GB** | **19.0 GB** | **20.9 GB** | 25.3 GB ⚠️ |
| `Qwen3.6-35B-A3B-4bit` | 21.3 GB | 21.8 GB | 22.4 GB | 23.6 GB |
| `gemma-4-31b-it-4bit` | 21.2 GB | 25.0 GB | 29.8 GB ⛔ | — |
| `Qwen2.5-14B-Instruct-4bit` | 10.5 GB | 15.3 GB ⚠️ | ⛔ 32k is its hard max | — |

> The Qwen2.5-14B row is not a prediction: the 32k figure was **measured at 15.225 GB** on this Mac
> (predicted 15.3 GB). The formula in this section reproduces reality to within 0.5%.

### 2.5 LoRA / QLoRA training headroom

LoRA adapter weights and their Adam states are **negligible** — measured below at 1.5–2.9 M
trainable params (0.3–0.6% of the model). The real cost is **backprop activations**, which scale as
`batch_size × max_seq_length × hidden_size × num_trainable_layers`.

**Measured on this Mac** (`mlx_lm.lora`, `Qwen/Qwen2.5-0.5B-Instruct` bf16, 494 M params,
~0.99 GB weights, batch 4, max-seq 1024, 20 iters on a synthetic transaction-categorization set):

| `--num-layers` | Trainable params | **Peak mem** | Overhead over weights | Throughput |
|---:|---|---:|---:|---:|
| 8 | 1.47 M (0.297%) | **1.679 GB** | +0.69 GB | 1,561 tok/s |
| 16 | 2.93 M (0.594%) | **1.827 GB** | +0.84 GB | 1,223 tok/s |

Doubling trainable layers cost only **0.148 GB** → **~18.5 MB per trained layer** at
batch 4 × seq 1024 × hidden 896. Normalising gives a scaling law:

```
activation_GB  ≈  2.5 × batch × seq_len × hidden_size × n_trained_layers × 2 bytes  +  0.5 GB
LoRA peak      ≈  base_weights  +  activation_GB
```

**Validation against independent third-party Apple Silicon runs** (mlx-lm defaults: batch 4,
`--num-layers 16`, seq 2048):

| Base model | Base weights | Formula predicts | Reported peak | Error |
|---|---:|---:|---:|---:|
| Qwen3.5-2B bf16 | 4.43 GB | 6.3 GB | **5.9 GB** | +7% |
| Qwen3.5-4B bf16 | 9.08 GB | 11.3 GB | **11.1 GB** | +2% |
| Qwen3.5-0.8B bf16 | 1.71 GB | 2.9 GB | 3.9 GB | −26% |
| Mistral-Nemo-12B 4-bit | ~6.9 GB | ~9 GB | 8.2 GB | +10% |

> **Rule of thumb: LoRA/QLoRA peak ≈ base weights + 1.5–3 GB** at mlx-lm defaults for models in the
> 0.5B–4B range; budget **+3–5 GB** at 9B and above, or if you raise `--max-seq-length`.
> Scaling knobs, in order of effect: `--max-seq-length` (linear in activations, quadratic in the
> attention term), `--batch-size` (linear), `--num-layers` (linear, and the cheapest to cut),
> `--grad-checkpoint` (trades ~30% more compute for a large activation saving).

Applied to this machine:

| Fine-tune target | Base | Est. LoRA peak | Verdict on 36 GB M3 Pro |
|---|---:|---:|---|
| Qwen3.5-0.8B bf16 | 1.73 GB | ~3–4 GB 🟢 | Trivial. Can train with a browser open. |
| **Qwen3.5-2B bf16** | 4.45 GB | **~6–7 GB** 🟢 | ✅ **Sweet spot.** Full-precision base, no quantization damage. |
| Qwen3.5-4B bf16 | 9.11 GB | ~11–12 GB 🟢 | ✅ Fine. |
| Qwen3.5-9B 4-bit (QLoRA) | 5.98 GB | ~9–11 GB 🟢 | ✅ Fine — you can QLoRA the workhorse too. |
| Qwen3.8-27B 4-bit (QLoRA) | 16.08 GB | ~20–24 GB 🔴 | ⚠️ Possible, but dedicates the machine. Don't. |

### 2.6 Speed rule of thumb

Generation is **bandwidth-bound**, not compute-bound:

```
tok/s  ≈  150 GB/s  ÷  (bytes read per token)  ×  ~0.75 efficiency
```

For a dense model, `bytes read per token ≈ full weight size`. For MoE, it is roughly the
**active** share.

| Model | Bytes/token | Predicted tok/s |
|---|---:|---:|
| `Qwen3.5-2B-4bit` | 1.75 GB | ~64 |
| `Qwen3.5-4B-MLX-4bit` | 3.06 GB | ~37 |
| `Qwen3.5-9B-4bit` | 5.98 GB | **~19** |
| `Qwen2.5-14B-Instruct-4bit` | 8.31 GB | ~13.5 |
| `gpt-oss-20b-MXFP4-Q8` (A3.6B) | ~2.5 GB | ~30–45 |
| `Qwen3.6-35B-A3B-4bit` (A3B) | ~2.5 GB | ~30–45 |
| `Qwen3.8-27B-4bit` | 16.05 GB | **~7** |

### 2.7 The budget ladder — "can I run model X here?"

| Zone | Total MLX peak | What it means |
|---|---|---|
| 🟢 **Green** | **≤ 14 GB** | Runs with browser, IDE, Slack open. No thought required. |
| 🟡 **Amber** | **≤ 22 GB** | Close the browser and heavy apps first. Fine for batch jobs. |
| 🔴 **Red** | **≤ 27 GB** | Dedicated run only. Quit everything. |
| ⛔ **Never** | **> 30.15 GB** | Exceeds `max_recommended_working_set_size`. Swap thrash; 10–50× slowdown. |

Decision procedure:

1. `weights ≈ params_B × 0.60` (4-bit) or `× 1.10` (8-bit) or `× 2.0` (bf16)
2. `kv ≈ ctx_tokens × KB_per_token` from §2.3 (or compute it with the §2.2 formula)
3. `+ 0.7 GB` workspace; `+ 2–4 GB` more if training
4. Compare against the ladder above.

If you must push into Red, raise the wired limit first:
`sudo sysctl iogpu.wired_limit_mb=30000`.

---

## 3. Recommendations by role

### Role 1 — General instruction / reasoning workhorse

| | **PRIMARY** | Alternative |
|---|---|---|
| **Repo** | `mlx-community/Qwen3.5-9B-4bit` | `mlx-community/gemma-4-12B-it-qat-4bit` |
| **Verified** | ✅ HF API, 5.98 GB | ✅ HF API, 11.02 GB |
| **Params** | 9.41 B (dense) | 11.96 B (dense) |
| **Resident** | ~6.0 GB | ~11.0 GB |
| **Peak @32k** | 7.8 GB 🟢 | 12.4 GB 🟢 |
| **Context** | 262,144 | 262,144 |
| **License** | Apache-2.0 | Apache-2.0 (Gemma 4 is Apache now, ungated) |
| **Speed** | ~19 tok/s | ~10 tok/s |

**Qwen3.5-9B — good at:** synthesis and narrative over long financial text; 262k native context with
a KV cache 6× cheaper than any same-size classic transformer; hybrid thinking mode you can switch
off per-request for cheap categorization and on for reconciliation reasoning; strong instruction
following and tool-calling; **natively multimodal**, so the same weights double as your vision model
under `mlx-vlm` (see Role 5). MMLU-Pro **82.5** (third-party, Qwen-reported).

**Bad at:** thinking mode burns a lot of output tokens — at ~19 tok/s a long chain-of-thought is
genuinely slow, so disable thinking for high-volume paths. It is a generalist, not a numerics
engine: never let it do arithmetic you can do in pandas. `mlx-lm` loads it **text-only**
(PR #869 was explicitly "support qwen3.5 series w/o vision"); vision needs `mlx-vlm`.

**Tradeoff vs `gemma-4-12B-it-qat-4bit`:** the QAT Gemma is quantization-aware-trained, so its
4-bit weights lose less quality than post-training quantization — but the repo keeps MLPs at 8-bit,
making it **1.8× the memory (11.0 vs 6.0 GB) and about half the speed** for a model in the same
class. Gemma's 5:1 sliding-window pattern also gives it 64 KB/token vs Qwen's 32. Take Gemma only if
you measure Qwen3.5-9B's 4-bit quantization degrading your categorization accuracy.

> `mlx-community/Qwen3.5-9B-MLX-4bit` is a byte-identical-size (5.98 GB) parallel conversion of the
> same weights and works equally well. Either is fine.

### Role 2 — Small model to fine-tune (LoRA/QLoRA)

| | **PRIMARY** | Alternative | Conservative fallback |
|---|---|---|---|
| **Repo** | `mlx-community/Qwen3.5-2B-bf16` | `mlx-community/Qwen3.5-4B-MLX-4bit` | `mlx-community/Qwen3-1.7B-bf16` |
| **Verified** | ✅ HF API, 4.45 GB | ✅ HF API, 3.06 GB | ✅ HF API, 3.46 GB |
| **Params** | 2.21 B | 4.54 B | 1.7 B |
| **Train peak** | ~6–8 GB 🟢 | ~7–9 GB 🟢 | ~5–7 GB 🟢 |
| **KV** | 12 KB/token | 32 KB/token | ~112 KB/token |
| **Arch** | `qwen3_5` (hybrid linear attn) | `qwen3_5` | `qwen3` (plain transformer) |

**Why bf16 for the fine-tune target, not 4-bit:** at 2B the full-precision base is only 4.45 GB, so
you have no reason to accept quantization error during training. Train in bf16, then quantize the
merged result with `mlx_lm.convert -q` for serving. QLoRA on a 4-bit base is the fallback for when
you move to 4B/9B.

**Qwen3.5-2B — good at:** it is the right size to actually *learn* a fixed schema. For transaction
categorization and field extraction you are teaching format compliance and a closed label set, not
world knowledge — a 2B converges fast (third-party: ~8 min for 600 iterations on Apple Silicon) and
serves at ~64 tok/s, which matters when you are grinding thousands of rows. Its KV cost of
12 KB/token is the lowest in the roster, so you can batch long CSV chunks cheaply. 262k context.

**Bad at:** zero-shot quality is mediocre — this model is *only* worth it once fine-tuned; do not
use it un-tuned as a fallback for Role 1. Its 248,320-token vocabulary is large relative to
2.2B params (embeddings are a big fraction of the weights), so it wastes some capacity on languages
you will never use.

**Tradeoff vs `Qwen3.5-4B-MLX-4bit`:** the 4B has meaningfully better zero-shot reasoning, so it
needs less training data to reach the same accuracy, and it still trains in ~11 GB. But it is ~1.7×
slower at inference and 2.6× the KV per token. **Start at 2B; escalate to 4B only if your eval gate
shows the 2B plateauing below target.**

**Tradeoff vs `Qwen3-1.7B-bf16` (the fallback):** Qwen3.5's gated-delta linear attention is newer
and less battle-tested under `mlx_lm.lora` than the plain `qwen3` transformer, which has 16 months
of community LoRA runs behind it. `mlx-lm`'s `linear_to_lora_layers()` auto-discovers every
`nn.Linear`/`QuantizedLinear` in `model.layers` and needs no per-architecture special-casing, so
Qwen3.5 *should* just work — and a third-party project reports successful LoRA on Qwen3.5-0.8B/2B/4B.
If you hit a training bug you cannot debug, drop to `Qwen3-1.7B-bf16` and lose only a little quality.

### Role 3 — Text embeddings for RAG

| | **PRIMARY** | Alternative | Rejected |
|---|---|---|---|
| **Repo** | `Qwen/Qwen3-Embedding-0.6B` | `mlx-community/Qwen3-Embedding-4B-4bit-DWQ` | `google/embeddinggemma-300m` |
| **Verified** | ✅ HF API, 1.21 GB | ✅ HF API, 2.28 GB | ✅ exists, but **gated** |
| **Dims** | 1024 (Matryoshka-truncatable) | 2560 | 768 |
| **Max ctx** | **32,768** | 32,768 | **2,048** ⛔ |
| **License** | Apache-2.0, **ungated** | Apache-2.0 | Gemma license, **manual approval required** |
| **Runtime** | `sentence-transformers` 6.x on MPS | `mlx_embeddings` / `mlx-vlm` | — |

**Qwen3-Embedding-0.6B — good at:** 32k context is the decisive property for financial RAG — you can
embed a whole statement page, a full transaction block, or a multi-page policy section without
shredding it into 512-token fragments that lose the account/period header. Instruction-aware
(prepend a task instruction to bias retrieval toward "find the transaction matching…"). Matryoshka
training means you can truncate 1024→256 dims and keep most of the quality, which shrinks your
vector store 4×. Apache-2.0 and ungated, so it survives HEARTH's sealed no-egress mode with no
license handshake. 6.8M downloads — the de facto default.

**Bad at:** 0.6B is a small encoder — on hard multi-hop retrieval it trails 4B/8B siblings. It is
also a decoder-derived embedder, so it is slower per document than a classic BERT-style encoder
(`all-MiniLM-L6-v2` is ~25× smaller and much faster if you only need cheap dense recall).

**Tradeoff vs `Qwen3-Embedding-4B-4bit-DWQ` (2.28 GB):** the 4B is measurably better on retrieval,
but it is 6.7× the params, ~4× slower to index, and 2560-dim vectors make your store 2.5× bigger.
For a personal financial corpus (thousands, not millions, of chunks) that trade is not worth it.
**Use 0.6B for the index; if recall is your bottleneck, add a reranker instead of a bigger embedder.**

**Reranker (recommended addition):** `mlx-community/Qwen3-Reranker-0.6B-4bit` — ✅ verified to exist.
Retrieve top-50 with the 0.6B embedder, rerank to top-5. This buys more accuracy per GB than any
embedder upgrade.

**Why `google/embeddinggemma-300m` is rejected:** two disqualifiers. (1) Its
`max_position_embeddings` is **2,048** — verified from config — which forces aggressive chunking of
exactly the long statements you care about. (2) The repo is **`gated: manual`**, requiring HF
license acceptance and an authenticated download; that is friction you do not want in a sealed
offline pipeline. (The ungated MLX mirror `mlx-community/embeddinggemma-300m-8bit`, 0.37 GB, exists
if you ever want a tiny fast embedder — but the 2,048-token limit remains.)

**Also verified available:** `mlx-community/bge-m3-mlx-8bit` (multilingual, 8k ctx) if you want a
second opinion in an ensemble.

### Role 4 — Offline LLM judge for eval gates

| | **PRIMARY** | Alternative | Also considered |
|---|---|---|---|
| **Repo** | `mlx-community/Qwen3.8-27B-4bit` | `mlx-community/Qwen3.6-35B-A3B-4bit` | `mlx-community/gpt-oss-20b-MXFP4-Q8` |
| **Verified** | ✅ HF API, 16.08 GB | ✅ HF API, 20.43 GB | ✅ HF API, 12.10 GB |
| **Params** | 27.36 B dense | 35.11 B MoE, ~3 B active | 20.91 B MoE, ~3.6 B active |
| **Peak @32k** | **19.0 GB** 🟡 | 21.8 GB 🟡 | ~13.5 GB 🟢 |
| **Peak @64k** | **20.9 GB** 🟡 | 22.4 GB 🟡 | ~14.1 GB 🟢 |
| **KV** | 64 KB/token | 20 KB/token | ~24 KB/token |
| **Speed** | ~7 tok/s | ~30–45 tok/s | ~30–45 tok/s |
| **Released** | Aug 2026 | Apr 2026 | 2025 |
| **License** | Apache-2.0 | Apache-2.0 | Apache-2.0 |

**Qwen3.8-27B-4bit — good at:** it is the newest and strongest thing that comfortably fits your
~22 GB budget with real room for long context — **19.0 GB at 32k, 20.9 GB at 64k**, both inside
Amber. Dense 27B gives more consistent judgement than a sparse model of nominally larger size, which
is what you want from a gate that has to be *stable* run over run. It exposes a `reasoning_effort`
control (`low`/`medium`/`xhigh`), so you can dial deliberation per eval tier. 262k context means a
judge prompt can hold the source document, the model output, and a full rubric at once. Apache-2.0.

**Bad at:** **~7 tok/s.** This is genuinely slow. It is fine for an eval gate run nightly or per-PR
over tens of cases; it is unusable interactively. Its 64 KB/token KV is 2× Qwen3.5-9B's, so pushing
past 64k context moves you into Red (25.3 GB at 128k). Use `reasoning_effort: low` for pass/fail
rubric checks and reserve higher effort for tie-breaks.

**Tradeoff vs `Qwen3.6-35B-A3B-4bit`:** this is the real decision. The MoE is the *biggest* model
that technically fits, and being ~3B-active it is **4–6× faster to generate**. But: 20.43 GB of
weights are resident regardless of sparsity, which eats essentially your entire 22 GB budget before
any KV — leaving ~1.5 GB, enough for only ~75k tokens and nothing else on the machine. It is also a
generation older (Apr vs Aug 2026), and 4-bit quantization hurts MoE experts more than dense layers
because each expert sees less activation traffic. **Pick the dense 27B for judgement quality and
headroom; pick the 35B-A3B only if judge wall-clock becomes your bottleneck.**

**Tradeoff vs `gpt-oss-20b-MXFP4-Q8`:** at 12.10 GB with a 128-token sliding window its KV is
almost free, and at ~13.5 GB total it stays 🟢 Green — meaning it can run **concurrently with the
Qwen3.5-9B workhorse** (13.5 + 7.8 = 21.3 GB) without unloading anything. That is a real
architectural advantage for a CI-style gate. But it is a 2025 model and a weaker judge. **Consider
it as the "fast tier" gate, with Qwen3.8-27B as the "deep tier" for releases.**

### Role 5 — Vision model for scanned / image PDFs

| | **PRIMARY** | Alternative | Zero-download option |
|---|---|---|---|
| **Repo** | `mlx-community/dots.ocr-4bit` | `mlx-community/Qwen3-VL-8B-Instruct-4bit` | `mlx-community/Qwen3.5-9B-4bit` via `mlx-vlm` |
| **Verified** | ✅ HF API, 3.54 GB | ✅ HF API, 5.78 GB | ✅ (Role 1 weights) |
| **Params** | 3.04 B (exact) | ~8 B + vision tower | 9.41 B (exact) |
| **Peak @8k** | ~4.3 GB 🟢 | ~7 GB 🟢 | ~7 GB 🟢 |
| **License** | MIT | Apache-2.0 | Apache-2.0 |
| **mlx-vlm arch** | `dots_ocr` ✅ | `qwen3_vl` ✅ | `qwen3_5` ✅ |

**dots.ocr — good at:** it is *purpose-built for document parsing*, not general image chat — layout
detection plus text extraction in one pass, which is exactly the shape of "read this scanned bank
statement." It emits structured layout (tables, headers, reading order) rather than a prose
description, so downstream transaction parsing gets rows instead of paragraphs. At 3.54 GB it stays
🟢 Green and can co-reside with the workhorse (3.5 + 7.8 = 11.3 GB), letting you OCR a page and
reason about it without unloading models. MIT licensed.

**Bad at:** it is an OCR/layout specialist — it cannot answer semantic questions about the image
("is this a duplicate charge?"). Handwriting and very low-DPI scans remain hard for any 3B OCR model.
You will still want a deterministic text-PDF path (`pdfplumber`/`pypdf`) and reserve this for the
image-only fallback.

**Tradeoff vs `Qwen3-VL-8B-Instruct-4bit`:** the general VLM can both read *and* reason — ask it
"extract every debit over $100 as JSON" and it will do the whole job in one call, where dots.ocr
needs a second LLM pass. The cost is 1.6× the memory, ~2× slower, and generally *worse raw
transcription accuracy* than a dedicated OCR model on dense tabular scans. **Use dots.ocr when you
want faithful extraction; use Qwen3-VL-8B when you want one-shot question answering over a page.**

**The zero-download option worth knowing about:** `Qwen3.5-9B` is a **native vision-language model**
— its `config.json` contains a full `vision_config` (27-layer, 1152-hidden vision tower, patch 16),
and `mlx-vlm` 0.6.17 ships a `qwen3_5` implementation with `vision.py`. So the Role 1 weights you
already downloaded can serve vision under `mlx-vlm` at **zero extra disk**. `mlx-lm` will not do this
— it loads Qwen3.5 text-only by design. **Try this first; only download dots.ocr if its transcription
accuracy on your actual statements is insufficient.**

**Other verified OCR options** (all exist, sizes from HF API): `mlx-community/PaddleOCR-VL-1.6-4bit`
(**0.72 GB** — remarkably small, Aug 2026), `mlx-community/DeepSeek-OCR-2-6bit` (3.30 GB, highest
downloads of the OCR set), `mlx-community/GLM-OCR-4bit` (1.25 GB),
`mlx-community/olmOCR-2-7B-1025-mlx-4bit` (5.65 GB), `mlx-community/Nanonets-OCR2-3B-4bit` (3.09 GB).

---

## 4. Co-residency plans

Because MLX shares one unified pool, what matters is which combinations fit simultaneously.

| Plan | Models resident | Peak @32k ctx | Zone |
|---|---|---:|---|
| **Interactive** | Qwen3.5-9B + Qwen3-Embedding-0.6B + dots.ocr | 7.8 + 1.2 + 3.5 = **12.5 GB** | 🟢 Green |
| **Ingest / OCR batch** | dots.ocr + Qwen3-Embedding-0.6B | **4.7 GB** | 🟢 Green |
| **Fine-tuning** | Qwen3.5-2B LoRA (nothing else) | **~8 GB** | 🟢 Green |
| **Fast eval gate** | gpt-oss-20b + Qwen3.5-9B | 13.5 + 7.8 = **21.3 GB** | 🟡 Amber |
| **Deep eval gate** | Qwen3.8-27B alone | **19.0 GB** | 🟡 Amber |
| ⛔ **Do not** | Qwen3.8-27B + Qwen3.5-9B | 19.0 + 7.8 = **26.8 GB** | 🔴 Red — unload one |

The headline: **the entire day-to-day roster (workhorse + embedder + OCR) runs concurrently in
12.5 GB.** The judge is the only thing that requires clearing the deck, which is fine because it
runs occasionally.

---

## 5. The Qwen2.5-14B question

**You are downloading `mlx-community/Qwen2.5-14B-Instruct-4bit` as the presumed workhorse.
Stop it and take `mlx-community/Qwen3.5-9B-4bit` instead.** It loses on every axis:

| | `Qwen2.5-14B-Instruct-4bit` | `Qwen3.5-9B-4bit` | Winner |
|---|---|---|---|
| Released | Sep 2024 | Mar 2026 (Qwen3.5 gen) | 3.5 |
| Disk / resident | 8.32 GB | **5.98 GB** | 3.5 (−28%) |
| Max context | **32,768** | **262,144** (8×) | 3.5 |
| KV per token | 192 KB | **32 KB** (6× cheaper) | 3.5 |
| Peak @32k ctx | 15.3 GB | **7.8 GB** | 3.5 (−49%) |
| Predicted tok/s | ~13.5 | **~19** | 3.5 (+40%) |
| MMLU-Pro (third-party) | **51.2** | **82.5** | 3.5 (+61%) |
| Multimodal | No | Yes (native VLM) | 3.5 |
| Thinking mode | No | Yes, switchable | 3.5 |

It is **smaller, faster, 8× longer context, 6× cheaper KV, dramatically stronger, and multimodal.**
There is no scenario in this project where the Qwen2.5-14B is the better choice.

> The MMLU-Pro figures are **third-party/vendor-reported**, not measured here — treat the *magnitude*
> as directional. Every other row in that table is verified from live configs and the HF API.

**What about the Qwen2.5-Coder models you already have?** Keep
`Qwen2.5-Coder-7B-Instruct-4bit` (4.0 GB) and `Qwen2.5-Coder-14B-Instruct-4bit` (7.7 GB) — they are
still perfectly good at *code*, which is a different job. Just do not use them as the financial
workhorse. `Qwen/Qwen2.5-0.5B-Instruct` (953 MB) is worth keeping as a fast smoke-test fixture.

---

## 6. Disk budget

| Item | Size |
|---|---:|
| Already cached | 12.7 GB |
| + Qwen3.5-9B-4bit (Role 1) | 5.98 GB |
| + Qwen3.5-2B-bf16 (Role 2) | 4.45 GB |
| + Qwen3-Embedding-0.6B (Role 3) | 1.21 GB |
| + Qwen3-Reranker-0.6B-4bit | ~0.4 GB |
| + Qwen3.8-27B-4bit (Role 4) | 16.08 GB |
| + dots.ocr-4bit (Role 5) | 3.54 GB |
| **New downloads** | **~31.7 GB** |
| **Total footprint** | **~44 GB** of 237 GB free |

Non-issue. You could add `gpt-oss-20b-MXFP4-Q8` (12.1 GB) as the fast-tier judge and still be at
56 GB.

---

## 7. Toolchain upgrade required (BLOCKING)

Nothing in the Qwen3.5/3.6/3.8 or Gemma-4 families will load on your current install.

| Package | Installed | Required | Why |
|---|---|---|---|
| `mlx` | **0.32.0** ✅ | ≥ 0.32.0 | Already satisfies `mlx-lm` (≥0.31.2) and `mlx-vlm` (≥0.32.0) |
| `mlx-lm` | **0.29.1** ⛔ | **0.31.3** | `qwen3_5` arch added in **0.30.7** (PR #869, 2026-02-12); `gemma4` also newer than 0.29.1 |
| `transformers` | **4.57.2** ⛔ | **≥ 5.0.0** | `mlx-lm` 0.31.3 hard-requires `transformers>=5.0.0` |
| `sentence-transformers` | **5.1.2** ⛔ | **6.0.1** | 5.1.2 pins `transformers<5`; 6.0.1 requires `transformers>=5,<6`. Must move together. |
| `mlx-vlm` | not installed | **0.6.17** | Needed for Role 5. Requires `transformers>=5.14.0`, `mlx>=0.32.0` |

```bash
pip install -U 'mlx-lm==0.31.3' 'transformers>=5.14,<6' 'sentence-transformers==6.0.1' 'mlx-vlm==0.6.17'
```

**Order matters and this is a coordinated jump.** `transformers` 4.x → 5.x is a major version; do it
in a scratch venv and re-run HEARTH's test suite before promoting. `sentence-transformers` must move
in the same step or pip will backtrack on the `transformers` pin.

**Verified facts behind this table:**
- `mlx-lm` v0.30.7 release notes (2026-02-12): *"[MODEL] support qwen3.5 series w/o vision by @JJJYmmm in PR #869"*.
- `mlx-lm` GitHub issue #1136 ("Add Qwen3.5 architecture support", error `Model type qwen3_5 not supported`) is **closed**.
- `mlx_lm/models/` on `main` contains `qwen3_5.py`, `qwen3_5_moe.py`, `gemma4.py`, `gemma4_text.py`, `qwen3_vl.py`, `gpt_oss.py` — confirmed by directory listing (127 architectures).
- `mlx_vlm/models/qwen3_5/` contains `vision.py`, `gated_delta.py`, `language.py` — confirmed by directory listing.
- PyPI: `mlx-lm` latest **0.31.3**; `mlx-vlm` latest **0.6.17**; `transformers` latest **5.16.1**; `sentence-transformers` latest **6.0.1**; `mlx` latest 0.32.2.

**LoRA compatibility note:** `mlx_lm/tuner/utils.py::linear_to_lora_layers()` discovers targets
generically — it walks `model.layers` and converts every `nn.Linear`, `nn.QuantizedLinear`,
`SwitchLinear`, `QuantizedSwitchLinear`, and `nn.Embedding` it finds (or any module exposing
`.to_lora`). **There is no per-architecture allowlist**, so the stale "Mistral, Llama, Phi2,
Mixtral, Qwen2, Gemma, OLMo, MiniCPM, InternLM2" list in `LORA.md` does not restrict you — Qwen3.5
LoRA works.

---

## 8. Download commands

```bash
# Role 1 — workhorse
hf download mlx-community/Qwen3.5-9B-4bit

# Role 2 — fine-tune target (bf16 base; quantize after merging)
hf download mlx-community/Qwen3.5-2B-bf16

# Role 3 — RAG
hf download Qwen/Qwen3-Embedding-0.6B
hf download mlx-community/Qwen3-Reranker-0.6B-4bit

# Role 4 — judge
hf download mlx-community/Qwen3.8-27B-4bit

# Role 5 — try the zero-download path first (Role 1 weights under mlx-vlm);
#          download only if transcription accuracy is insufficient
hf download mlx-community/dots.ocr-4bit
```

---

## 9. Verification ledger

| Claim | How verified |
|---|---|
| Every repo id and its disk size | Live `GET huggingface.co/api/models/<id>?blobs=true`, summed blob bytes, 2026-09-02 |
| Layer counts, KV heads, head dim, context, licenses | Live `GET huggingface.co/<id>/raw/main/config.json` |
| `max_recommended_working_set_size` = 30.15 GB, `max_buffer_length` = 22.61 GB | `mx.device_info()` executed **on this Mac** |
| KV formula = 194.6 KB/token measured vs 192 predicted | `mlx_lm` generation run **on this Mac** across 256→32,000 tokens |
| Total footprint 15.225 GB @32k for the 14B (predicted 15.3) | Same run, `mx.get_peak_memory()` |
| LoRA activation scaling law (~18.5 MB per trained layer @ batch 4 × seq 1024 × hidden 896) | `mlx_lm.lora` run **on this Mac** on the cached `Qwen/Qwen2.5-0.5B-Instruct`, `--num-layers` 8 vs 16 |
| 4-bit = 0.563 GB/1B params | Measured resident memory (8.309 GB) of the locally cached 14B |
| `mlx-lm` arch support, LoRA targeting logic | GitHub contents API + raw source of `mlx_lm/models/`, `mlx_lm/tuner/utils.py` |
| Package versions and dependency pins | PyPI JSON API |
| `google/embeddinggemma-300m` is `gated: manual` | HF API `gated` field |
| MMLU-Pro scores, LoRA training peaks on Apple Silicon, M3 Pro = 150 GB/s | **Third-party / vendor-reported.** Directional only. |
| `mlx-community/Qwen3-Reranker-0.6B-4bit-DWQ` | ❌ **Does not exist** (404/401). The correct id is `mlx-community/Qwen3-Reranker-0.6B-4bit`. |
