# APEX × HEARTH — integration seam analysis

**Status:** Analysis, 2026-09-02. Read-only survey of two sibling repos on this machine.
**Scope:** `/Users/miltronix/Claude/apps/APEX` (finance domain) and `/Users/miltronix/Claude/apps/HEARTH` (model/infra layer).
**Framing:** HEARTH is the reusable model + infrastructure layer; APEX is the finance domain layer.
Dependency points one way — APEX may depend on HEARTH, never the reverse.

> **Privacy note on how this was produced.** No file under `APEX/FINANCES/` was opened, and no
> `.csv` / `.xlsx` / `.numbers` / `.pdf` holding transactions, balances, holdings, or account
> identifiers was read. Data formats below are described from the *parsing code*
> (`backend/processors/`), never from a data file. No real financial value appears in this
> document.

---

## Executive summary

| Question | Answer |
| --- | --- |
| Does APEX speak OpenAI today? | **No.** It speaks Ollama's *native* `/api/chat`, not `/v1`. |
| Is HEARTH a base-URL swap? | **No** — four concrete blockers, all small. See §2. |
| Effort to point APEX at HEARTH | **~1–2 days**, one new file + 5 one-line call-site edits. |
| Egress points found in APEX | **27**, enumerated in §4. |
| Egress from APEX's *finance* half | **Zero.** Every egress point lives under `backend/invest/`. |
| Third-party telemetry / analytics / crash reporting | **None.** Grep-clean. |
| Biggest duplication | Embeddings + vector store — **already literally copied from HEARTH**. |
| Biggest asymmetry | Document ingestion — APEX has a mature stack, **HEARTH has none at all**. |

**Prior art.** APEX has already specified this integration: `APEX/docs/ui/HEARTH_BRAIN.md`
(accepted spec) and `APEX/docs/ui/DECISIONS.md:84` (ADR-U004), with a terminal-runnable harness at
`APEX/scripts/hearth_offload_probe.py`. The conclusions below largely **confirm** that spec, with
four corrections in §2.4 where it is optimistic about the transport. Notably, HEARTH's own docs
(`docs/INTEGRATION.md`, `docs/RUNBOOK_consumer_wiring.md`) mention APEX **zero times** — the
dependency is documented on one side only.

---

## 1. How APEX talks to its LLM today

### 1.1 The provider abstraction exists and is good

APEX is **not** a codebase with Ollama calls scattered through it. There is exactly one place in
the entire repo that opens a socket to a model, and it sits behind a `Protocol`.

| Layer | File | What it is |
| --- | --- | --- |
| Contract | `APEX/backend/llm/client.py:27` | `LLMClient` Protocol — `complete(system, user, *, json=False) -> str`. That is the whole interface. |
| Local impl | `APEX/backend/llm/client.py:138` | `OllamaClient` |
| Remote impl | `APEX/backend/llm/client.py:34` | `AnthropicClient` (lazy SDK import) |
| Privacy gate | `APEX/backend/llm/router.py:29` | `LLMRouter` — routes on a required `DataClass` |
| Budget | `APEX/backend/llm/budget.py:46` | `TokenBudget` — per-UTC-day remote cap |
| Assembly | `APEX/backend/llm/router.py:93` | `build_router()` |

The invest subsystem's own LLM package is a **pure re-export shim** — `APEX/backend/invest/llm/client.py:7`
and `APEX/backend/invest/llm/router.py:7` both just `from ...llm.X import ...` for backward
compatibility. There is no second implementation.

### 1.2 The call interface — Ollama *native*, not OpenAI

`OllamaClient.complete` (`APEX/backend/llm/client.py:165-212`) posts to
**`{host}/api/chat`** (`client.py:202`) — Ollama's proprietary endpoint, not
`/v1/chat/completions`. The body (`client.py:192-199`):

```python
{"model": ..., "messages": [...], "stream": False,
 "options": {"temperature": ..., "num_predict": ...},   # Ollama-specific
 "format": "json"}                                       # Ollama-specific, only when json=True
```

Response is read as `data["message"]["content"]` (`client.py:208`) — Ollama's shape, not
`choices[0].message.content`.

Transport is `httpx.post` (`client.py:202`), timeout 600s (`client.py:153`), no `ollama` SDK
despite `ollama==0.1.6` still sitting in `APEX/backend/requirements.txt:11`.

> **Dead config, worth deleting.** `APEX/.env.example:5` declares
> `OLLAMA_BASE_URL=http://localhost:11434/v1` (note the `/v1`), and the key is present in the live
> `.env`. **Nothing in the codebase reads it** — verified by grep across all `.py` and `.swift`.
> The real knob is `settings.ollama_host = "http://localhost:11434"` (`APEX/backend/config.py:33`),
> with **no** `/v1`. Anyone reading `.env.example` will wrongly conclude APEX is already
> OpenAI-shaped.

### 1.3 Models referenced

| Setting | Default | file:line |
| --- | --- | --- |
| `ollama_model` | `qwen2.5:7b` — general / finance RAG | `APEX/backend/config.py:34` |
| `invest_llm_model` | `qwen2.5:32b` — the trading analyst | `APEX/backend/config.py:92` |
| `embedding_model` | `sentence-transformers/all-MiniLM-L6-v2` (Qdrant path only) | `APEX/backend/config.py:35` |
| `AnthropicClient.DEFAULT_MODEL` | `claude-opus-5` | `APEX/backend/llm/client.py:59` |

`OllamaClient` defaults to `invest_llm_model` (the 32B), **not** `ollama_model`
(`client.py:155`) — so a bare `OllamaClient()` gets the heavy model. `LLMService` overrides it to
the 7B for finance RAG (`APEX/backend/rag/llm.py:29,34`).

### 1.4 The data-class boundary — the thing that actually matters

`LLMRouter.complete` (`APEX/backend/llm/router.py:58`) **requires** a `DataClass` and raises
without one (`router.py:60-61`). `DataClass.PERSONAL` returns at `router.py:74-76` before any
remote branch is reachable, with the comment *"NEVER remote, regardless of config/budget."*
The boundary is structural, not a convention.

Every call site, exhaustively:

| Call site | DataClass | file:line |
| --- | --- | --- |
| Finance agent — plan | PERSONAL | `APEX/backend/finance/agent.py:365` |
| Finance agent — synthesize | PERSONAL | `APEX/backend/finance/agent.py:408` |
| Finance RAG generate | PERSONAL | `APEX/backend/rag/llm.py:64` |
| Research claim drafting | PUBLIC | `APEX/backend/invest/research/agent.py:130` |
| Research council (5 lenses) | PUBLIC | `APEX/backend/invest/research/council.py:155` |

Both PUBLIC sites were audited. `council.py:143-147` builds its prompt from the claim text plus
rendered public evidence only; `agent.py:127` builds `SYMBOLS OF INTEREST: … EVIDENCE: …`. The
symbol list defaults to `SPRINT_ALLOWLIST` — a static published allowlist, not the live book
(`APEX/backend/invest/cli.py:1265`). **No portfolio, cash, or position data reaches either.**
The classification is honest.

### 1.5 Coupling verdict

**Loosely coupled.** Ollama appears in exactly **five** construction sites:

| # | file:line | Context |
| --- | --- | --- |
| 1 | `APEX/backend/llm/router.py:118` | `build_router()` — the app-wide default |
| 2 | `APEX/backend/finance/agent.py:427` | `build_finance_agent()` fallback |
| 3 | `APEX/backend/invest/service.py:181` | Live trading daemon's analyst |
| 4 | `APEX/backend/invest/cli.py:1266` | `apex invest research --local-only` |
| 5 | `APEX/backend/rag/llm.py:34` | Finance RAG `LLMService` |

Everything else takes an injected `LLMClient` or `LLMRouter`. `LLMAnalystStrategy.llm` is a plain
dataclass field (`APEX/backend/invest/strategies/llm_analyst.py:66`); `FinanceAgent.router` is a
dataclass field (`APEX/backend/finance/agent.py:314`). This is close to the best case for a
provider swap.

**One caveat.** `LLMAnalystStrategy.propose` calls `self.llm.complete(...)` **directly**
(`llm_analyst.py:85`), bypassing `LLMRouter` entirely. It is safe today only because
`service.py:181` hands it a concrete `OllamaClient()`. The type is `LLMClient`
(`llm_analyst.py:66`) — nothing structurally prevents a future caller from injecting a remote
client and shipping the portfolio off-machine. The class docstring asserts the invariant
(`llm_analyst.py:9`, `:50`) but the code does not enforce it. **Recommend:** route it through
`LLMRouter` with `data_class=DataClass.PERSONAL` like every other personal call site.

---

## 2. What it takes to point APEX at HEARTH

### 2.1 Verdict: not a base-URL swap — but close

The premise ("Ollama has an OpenAI-compatible endpoint, so this may be nearly a `base_url` swap")
does not hold, for two independent reasons:

1. **APEX does not use Ollama's OpenAI endpoint.** It uses native `/api/chat`
   (`APEX/backend/llm/client.py:202`) with Ollama-proprietary `options` and `format` fields.
   There is no OpenAI-shaped client in APEX to re-point. The only OpenAI SDK usage in the entire
   tree is in dead legacy code: `APEX/legacy/apex-standalone/clients/ollama_client.py:3`.
2. **HEARTH's `/v1` is OpenAI-*shaped*, not OpenAI-*complete*.** Four concrete gaps, below.

### 2.2 The four blockers

**(a) Bearer auth is mandatory.** HEARTH's `require_auth` defaults `True`
(`HEARTH/src/hearth/config.py:32`); `require_token` (`HEARTH/src/hearth/gateway/auth.py:21`)
enforces `Authorization: Bearer <token>` on every route except the two admin probes. The token is
auto-generated to `~/.hearth/token` mode 0600 (`HEARTH/src/hearth/config.py:103`). Ollama needs no
header. APEX must read and send the token — and must handle its absence.

**(b) No JSON mode — this is the real work.** HEARTH's `ChatCompletionRequest`
(`HEARTH/src/hearth/gateway/schemas.py:28`) has six fields and no `model_config`, so pydantic's
default `extra="ignore"` silently drops everything else — **including `response_format`**.
There is no structured-output constraint of any kind.

APEX depends on constrained JSON at four call sites, all of which parse the reply:

| Call site | Consequence of a malformed reply |
| --- | --- |
| `APEX/backend/invest/strategies/llm_analyst.py:85` | Trade candidates dropped for the tick (`llm_analyst.py:90-93`) |
| `APEX/backend/finance/agent.py:365` | Plan returns `[]` — agent answers with no tools |
| `APEX/backend/invest/research/agent.py:130` | No claims drafted this run |
| `APEX/backend/invest/research/council.py:155` | **Counted as a refutation** (`council.py:126-131`) |

Today `json=True` becomes Ollama's `"format": "json"` (`client.py:198-199`), a grammar-level
constraint. Against HEARTH that becomes *prompt instruction only* — the same degraded approach
`AnthropicClient` already takes (`APEX/backend/llm/client.py:99-107`). The failure is soft and
fail-closed everywhere, so this is a **quality regression, not a safety one** — but the research
council in particular will refute more claims purely on parse failures.

**(c) `max_tokens` defaults to 512.** `HEARTH/src/hearth/gateway/schemas.py:28` sets
`max_tokens: int = 512`; OpenAI's default is model-max, and Ollama's `num_predict` is unset by
default in APEX (`client.py:184-186`). A client that omits it gets silently truncated — and
`finish_reason` is **hardcoded `"stop"`** (`HEARTH/src/hearth/gateway/schemas.py:56`), so
truncation is invisible. `LLMAnalystStrategy`'s JSON candidate list and `FinanceAgent._synthesize`
both routinely exceed 512 tokens. **APEX must send `max_tokens` explicitly.**

**(d) `model` is never validated.** `Router._local_model`
(`HEARTH/src/hearth/router/route.py:292`) passes any non-`"auto"` string through verbatim; the
gateway never calls `Registry.resolve()`. Sending `qwen2.5:32b` (APEX's Ollama tag) yields an echo
from the echo backend or a `503 hearth.provider.unavailable` from MLX — never a clean
`model_not_found`. **APEX should send `"auto"`** and let HEARTH's registry decide
(`HEARTH/config/models.yaml`).

Lesser gaps, non-blocking for APEX: `/v1/completions` is a 404; `messages[].content` must be a
plain string (a list of content parts is a 422); `role: "tool"`/`"function"`/`"developer"` is a
422; no tool/function calling; `seed`, `top_p`, `stop`, `n`, `logprobs` all silently ignored.
None of these are on any APEX code path.

### 2.3 File-by-file change list

**New — `APEX/backend/llm/hearth_client.py`** (~130 lines). A `HearthClient` mirroring
`OllamaClient`'s constructor shape and satisfying the same `LLMClient` Protocol:

- `POST {base}/v1/chat/completions` via `httpx` (already a dependency, `requirements.txt:44`).
- Token from `HEARTH_TOKEN` env, else `~/.hearth/token`, sent as `Authorization: Bearer`.
- `(system, user)` → `messages=[{role:"system"},{role:"user"}]`; skip the system turn when empty,
  matching `client.py:187-190`.
- `model="auto"`; **always send an explicit `max_tokens`** (default ≥ 2048).
- `json=True` → append the same instruction `AnthropicClient` uses (`client.py:106-107`).
- Send the HEARTH extension block `{"intent": ..., "allow_escalation": False}` — this is a
  hard local pin at HEARTH's router (`HEARTH/src/hearth/router/route.py:116`), giving
  defence-in-depth beneath APEX's own `DataClass` gate.
- Parse `choices[0].message.content`; raise `LLMError` on anything else, so the existing
  degradation paths (`llm_analyst.py:87`, `agent.py:403`) work unchanged.
- Add a **loopback guard** copied from `APEX/scripts/hearth_offload_probe.py:257-262`. Refuse a
  non-loopback base URL, fail-closed. See §4.5 — this is the single guard `OllamaClient` lacks.
- `name` property returning `f"hearth:{model}"`, matching `client.py:161-163`.
- Optionally surface `hearth.estimated_frontier_tokens_saved` from the response
  (`HEARTH/src/hearth/gateway/schemas.py:37`) to corroborate `TokenBudget.record_local`.

**New — `FallbackClient`** (~25 lines, same file). `primary=HearthClient, secondary=OllamaClient`;
catch the primary's `LLMError` and retry the secondary. `HEARTH_BRAIN.md` recommends this and it
is the difference between "HEARTH outage skips a trading tick" and "HEARTH outage is invisible."

**Edit — `APEX/backend/config.py`** (~6 lines, after line 37). Add `hearth_base_url:
str = "http://127.0.0.1:8080"`, `hearth_enabled: bool = False`, `hearth_model: str = "auto"`,
`hearth_max_tokens: int = 2048`. Default **off**, per ADR-U004's "never a hard dependency."

**New — `select_local_client()` factory** in `APEX/backend/llm/__init__.py`. Returns
`FallbackClient(HearthClient(), OllamaClient())` when `settings.hearth_enabled`, else
`OllamaClient()`. Export it alongside the existing names (`__init__.py:11-18`).

**Edit — the five construction sites**, one line each:

| # | file:line | Change |
| --- | --- | --- |
| 1 | `APEX/backend/llm/router.py:118` | `local = select_local_client()` |
| 2 | `APEX/backend/finance/agent.py:427` | `LLMRouter(local=select_local_client())` |
| 3 | `APEX/backend/invest/service.py:181` | `llm=select_local_client()` |
| 4 | `APEX/backend/invest/cli.py:1266` | `LLMRouter(local=select_local_client())` |
| 5 | `APEX/backend/rag/llm.py:34` | `select_local_client(model=..., temperature=...)` |

**Edit — `APEX/backend/invest/provenance.py:107`.** The config-provenance stamp records
`invest_llm_model`. Add the HEARTH knobs, or a live A/B spanning the switch becomes
uninterpretable — exactly the 2026-07-30 failure that module's comment documents
(`APEX/backend/invest/service.py:196-198`).

**Edit — `APEX/.env.example:5`.** Delete the dead `OLLAMA_BASE_URL`; add `APEX_HEARTH_ENABLED`.

**New — `APEX/tests/backend/test_hearth_client.py`.** Mirror the existing
`tests/backend/test_llm_layer.py` and `test_llm_remote.py` patterns: fake transport, assert
message mapping, assert `max_tokens` is always sent, assert the loopback guard rejects a remote
host, assert `FallbackClient` falls through on `LLMError`.

**No change needed:** every consumer takes an injected `LLMClient`/`LLMRouter`. The `DataClass`
boundary, the `TokenBudget`, the three MCP servers, and the whole finance stack are untouched.

### 2.4 Corrections to APEX's existing HEARTH_BRAIN.md spec

The spec is sound. Four points to amend:

1. §2 calls APEX's client *"already OpenAI-shaped in spirit."* It is Ollama-native
   (`client.py:202`). The mapping is still small, but it is a rewrite, not a re-point.
2. The spec does not mention **auth**. HEARTH requires a bearer token by default
   (`HEARTH/src/hearth/config.py:32`).
3. The spec does not mention **JSON-mode loss**, which is the largest behavioural delta (§2.2b).
4. The spec does not mention **`max_tokens: 512`** (§2.2c), which will silently truncate the
   analyst and the finance agent.

### 2.5 Effort estimate

| Task | Estimate |
| --- | --- |
| `HearthClient` + `FallbackClient` + loopback guard | 3–4 h |
| Config, factory, five call sites, provenance, `.env.example` | 1 h |
| Unit tests (fake transport, no network) | 2 h |
| JSON-reliability hardening: retry-once-on-parse-failure, fence stripping | 2–3 h |
| Live validation on real hardware (`scripts/hearth_offload_probe.py`, then a paper tick) | 2 h |
| **Total** | **~1–2 working days** |

**Risk: low.** Nothing on the privacy boundary changes — HEARTH is loopback, so PERSONAL work
routed to it never crosses the machine boundary, and `LLMRouter`'s PERSONAL branch
(`router.py:74-76`) is untouched. Every consumer already degrades gracefully on `LLMError`.
Default-off means the change is inert until explicitly enabled.

**The one thing to watch:** the research council's refutation rate. It counts an unparseable
reply as a refutation (`council.py:126-131`), so losing grammar-constrained JSON will make the
council *more* conservative. That is fail-safe, but measure it — a council that refutes everything
looks identical to a council that is working.

---

## 3. What is duplicated, and who should own it

### 3.1 The headline: two modules are already literal copies

This is not conceptual overlap. APEX's docstrings say so outright.

| APEX module | Says | HEARTH origin |
| --- | --- | --- |
| `APEX/backend/knowledge/embed.py:3` | *"Ported from HEARTH (`src/hearth/memory/embed.py`), de-branded"* | `HEARTH/src/hearth/memory/embed.py` |
| `APEX/backend/knowledge/store.py:3` | *"Ported from HEARTH (`src/hearth/memory/store.py`), de-branded"* | `HEARTH/src/hearth/memory/store.py` |
| `APEX/backend/llm/budget.py:5` | *"Ported from HEARTH (`src/hearth/observability/budget.py`)"* | `HEARTH/src/hearth/observability/budget.py` |
| `APEX/app/APEX/Sources/Inference/OnDeviceInference.swift` | *"de-branded from HEARTH's `FoundationModelsProvider`"* (per `HEARTH_BRAIN.md` §1) | HEARTH Swift package |

Both embedders are the same 256-dim signed-hashing-trick design over blake2b
(`APEX/backend/knowledge/embed.py:41-45`; `HEARTH/src/hearth/memory/embed.py:48`). Both stores are
one-SQLite-file-per-collection with brute-force cosine and optional numpy
(`APEX/backend/knowledge/store.py:1-9`; `HEARTH/src/hearth/memory/store.py:57`).

### 3.2 Ownership recommendations

| # | Overlap | APEX | HEARTH | Owner | Why |
| --- | --- | --- | --- | --- | --- |
| 1 | **Embeddings** | `backend/knowledge/embed.py` (hash); `backend/rag/embeddings.py:24` (sentence-transformers) | `src/hearth/memory/embed.py` (hash + MLX) | **HEARTH** | Same code already. HEARTH's is strictly ahead — it has an `MLXEmbedder` (`embed.py:88`) and a plugin entry-point group (`embed.py:152`) that APEX lacks. Embedding quality is a model-layer concern, not a finance concern. |
| 2 | **Vector store** | `backend/knowledge/store.py` (SQLite); `backend/rag/vector_store.py:33` (Qdrant) | `src/hearth/memory/store.py` (SQLite + sqlite-vec) | **HEARTH** | Same code already. HEARTH additionally has a real KNN path via `SqliteVecVectorStore` (`store.py:184`). |
| 3 | **Qdrant stack** | `backend/rag/{engine,embeddings,vector_store}.py` + `docker-compose.yml:5` | — | **Retire from APEX** | Already de-facto dead: `APEX/backend/rag/factory.py:26-28` defaults to `knowledge`, and `knowledge_rag.py:1-9` calls the Qdrant path *"the only outbound network call left in the finance half."* It costs torch + sentence-transformers + a daemon + a HuggingFace download for a capability HEARTH provides locally. Delete it, or leave it strictly behind `APEX_RAG_BACKEND=qdrant` as today. |
| 4 | **Token budget** | `backend/llm/budget.py` | `src/hearth/observability/budget.py` | **HEARTH owns the design; APEX keeps a vendored copy** | Do *not* make APEX import it. ADR-U004's conformance rule is "APEX runs with HEARTH absent," and a shared budget module would break that. Copy-forward is correct here; the ~120 lines are stable. |
| 5 | **Router / policy** | `backend/llm/router.py` (`DataClass` PERSONAL/PUBLIC) | `src/hearth/router/` (task class + escalate policy) | **Both — do NOT merge** | These are different axes. APEX's is a **privacy** boundary (does this touch the user's money?); HEARTH's is a **cost/capability** boundary (is this worth a frontier token?). Keep APEX's as the outer gate; HEARTH's router runs *inside* the local leg. Merging them would put a privacy invariant in a repo that does not know what a portfolio is. |
| 6 | **Model management** | Two string knobs: `config.py:34`, `config.py:92` | `src/hearth/registry/` + `registry/adapters.py` + `config/models.yaml` | **HEARTH** | Not a close call. HEARTH has a versioned catalog, a candidate→promoted→retired adapter lifecycle, and an eval gate that *raises* `GateNotPassedError` without a promotion proof (`HEARTH/src/hearth/registry/adapters.py:117`). APEX has two hardcoded Ollama tags. Once wired, APEX should send `model: "auto"` and delete its knobs. |
| 7 | **RAG chunk/query loop** | `backend/rag/knowledge_rag.py`, `backend/knowledge/base.py` | `src/hearth/memory/rag.py` | **HEARTH owns generic; APEX keeps finance-specific** | The chunking, embedding and retrieval are generic → HEARTH. What is *not* generic is APEX's transaction→text rendering, batched at `_TXN_BATCH = 10` — *"one chunk ≈ one statement page's worth of line items"* (`APEX/backend/rag/knowledge_rag.py:36-38`). That is domain knowledge; it stays in APEX. |
| 8 | **MCP servers** | Three: `mcp-server/server.py`, `backend/finance/mcp_server.py`, `backend/invest/mcp_server.py` | One: `src/hearth/mcp/server.py` | **Both — no consolidation** | **Zero tool-name collisions and zero semantic overlap.** APEX exposes domain tools — `propose_trades`, `explain_position`, `portfolio`, `audit_summary`, `search_lessons`, `backtest` (`APEX/backend/invest/mcp_server.py:31-77`), `finance_agent`, `subscriptions`, `anomalies`, `categorize` (`APEX/backend/finance/mcp_server.py:50-71`). HEARTH exposes generic NL primitives — `hearth_summarize`, `hearth_classify`, `hearth_extract`, `hearth_draft`, `hearth_rag_query` (`HEARTH/src/hearth/mcp/server.py:26-46`). Different layers, correctly separated. Both are stdio. |
| 9 | **Document ingestion** | `backend/processors/` — mature | **None** | **APEX** | Not duplication — a gap. See §5. |

### 3.3 The one structural change worth making

Overlaps 1, 2 and 7 all resolve the same way, and there is a clean seam for it: once APEX speaks
to HEARTH over HTTP, APEX's `KnowledgeRAG` can delegate to `POST /v1/hearth/rag/query`
(`HEARTH/src/hearth/gateway/app.py:251`) and `POST /v1/embeddings`
(`HEARTH/src/hearth/gateway/app.py:153`) instead of running its own copies.

**But not yet, and one HEARTH change is required first** — see §5.3. `POST /v1/hearth/rag/ingest`
(`app.py:239`) takes `paths: list[str]` — *server-side filesystem paths*. There is no
text-payload ingest and no upload endpoint anywhere in HEARTH. APEX cannot hand HEARTH the text it
extracted from a bank PDF without writing it to a temp file and passing a path, which is both ugly
and a new place for financial data to land on disk unencrypted.

Until HEARTH grows a text-payload ingest, APEX's vendored copies of `embed.py`/`store.py` are the
right call: they cost nothing at runtime, keep the "runs with HEARTH absent" guarantee, and the
code is already written.

---

## 4. Where APEX's data actually flows — the egress inventory

**This is the most important section.** Every network call in APEX was enumerated by grepping all
literal URLs and all `httpx` / `requests` / `urllib` / `aiohttp` / SDK call sites across
`backend/`, `cli/`, `app/`, `ios/`, `mcp-server/`, `scripts/` and `container/`.

### 4.1 The headline finding

> **The finance half of APEX has zero egress.**

`backend/finance/`, `backend/processors/`, `backend/knowledge/`, `backend/security/` and
`backend/rag/knowledge_rag.py` contain **no** network call of any kind. The only sockets reachable
from a finance code path are:

- the local model on `settings.ollama_host`, default `http://localhost:11434`
  (`APEX/backend/config.py:33`), and
- Qdrant on `localhost:6333` (`APEX/backend/config.py:40-41`), only under
  `APEX_RAG_BACKEND=qdrant`.

`APEX/backend/rag/knowledge_rag.py:8` states the design intent: *"no daemon, no download, no
socket."* The code matches the claim.

**Everything below lives under `backend/invest/` (trading), plus one optional LLM client and one
optional model download.** Not one of these paths is reachable from `apex finance`.

### 4.2 Egress that can carry PERSONAL financial data

| # | file:line | Destination | What leaves | Gate |
| --- | --- | --- | --- | --- |
| **1** | `APEX/backend/llm/client.py:109` | `api.anthropic.com` | Prompt + system text | **Structurally gated.** `LLMRouter` returns at `router.py:74-76` for PERSONAL before any remote branch. Both live PUBLIC sites audited clean (§1.4). |
| **2** | `APEX/backend/llm/router.py:123-124` | `api.anthropic.com` | Nothing (credential probe) | ⚠️ **`build_router()` probes credentials by *constructing the client*, not by reading `ANTHROPIC_API_KEY`** — deliberately, per `router.py:105-108`, because the SDK also resolves `ANTHROPIC_AUTH_TOKEN` and an `ant auth login` profile. **No `ANTHROPIC_API_KEY` is set in APEX's `.env`, yet remote can still silently activate** on a machine where `ant auth login` has been run. Documented, intentional — but it means "no key in .env" is not evidence of no remote. |
| **3** | `APEX/backend/invest/broker/alpaca.py:203` | `api.alpaca.markets` / `paper-api.alpaca.markets` (`config.py:68-69`) | **Account equity, cash, every position, every order.** `portfolio_snapshot` (`alpaca.py:263`), `positions` (`:281`), `submit` (`:463`), `close_position` (`:537`), `closed_fills` (`:438`) | None, and none possible — this *is* the brokerage. Personal by construction. |
| **4** | `APEX/backend/invest/broker/alpaca.py:582` | Alpaca websocket | Live fill stream — symbol, qty, price, per fill | Same. |
| **5** | `APEX/backend/invest/broker/kalshi.py:102` | `api.elections.kalshi.com` / `demo-api.kalshi.co` (`kalshi.py:46-47`) | Authenticated (RSA-PSS/SHA256) event-contract activity | Same. |
| **6** | `APEX/backend/invest/authority/apns.py:191` | `api.push.apple.com` / `api.sandbox.push.apple.com` (`apns.py:32-33`, URL built `apns.py:129-131`) | ⚠️ **`build_payload` (`apns.py:97-115`) sends `symbol`, `side`, `notional` (a dollar figure), and the free-text `rationale`** to Apple, unencrypted-at-rest-on-Apple's-side. This is real personal trading data crossing the machine boundary to a third party. | **Opt-in.** `build_notifier()` (`apns.py:206-211`) returns `LoggingNotifier` unless every `APEX_APNS_*` var is set. None are set in the current `.env`. |

### 4.3 Egress carrying tickers / queries (public data, but the *query* is revealing)

These fetch public information. The leak is not content — it is that a third party learns which
symbols you are researching. The default symbol set is a published static allowlist
(`SPRINT_ALLOWLIST`, `APEX/backend/invest/cli.py:1265`), not the live book, which materially
limits this.

| # | file:line | Destination |
| --- | --- | --- |
| 7 | `APEX/backend/invest/broker/alpaca.py:210`, `:214` | Alpaca market data (stocks, crypto) |
| 8 | `APEX/backend/invest/broker/alpaca.py:218`, `:346` | Alpaca news API, **per-symbol** |
| 9 | `APEX/backend/invest/broker/polymarket.py:126` | `gamma-api.polymarket.com` (`:32`) — unauthenticated reads per `:6` |
| 10 | `APEX/backend/invest/strategies/cross_venue.py:289` | `api.elections.kalshi.com` (`:49`) |
| 11 | `APEX/backend/invest/feeds/macro.py:82` | `api.stlouisfed.org` via `fredapi` — **`FRED_API_KEY` is set in `.env`** |
| 12 | `APEX/backend/invest/feeds/spot_history.py:69` | `api.exchange.coinbase.com` (`:44`) |
| 13 | `APEX/backend/invest/feeds/weather.py:211+` | `api.weather.gov` (`:34`), `api.elections.kalshi.com` (`:33`) |
| 14 | `APEX/backend/invest/research/websearch.py:100` | Fed + SEC press RSS (`:50-51`) |
| 15 | `APEX/backend/invest/research/websearch.py:305` | ⚠️ **Arbitrary third-party URLs** — `fetch_article_parts` downloads any article URL that appeared in a feed. Bounded by `MAX_DOWNLOAD_BYTES` and a content-type check, but the host set is not fixed. Outbound only; nothing personal is sent. |
| 16 | `APEX/backend/invest/research/websearch.py:483` | ⚠️ `api.tavily.com/search` (`:456`) — **sends a constructed query string** (`:474`) **and your API key in the JSON body** (`:477`). **Disabled unless `RESEARCH_SEARCH_API_KEY` is set** (`:468-470`); it is not set. The class docstring (`:452-459`) explains the deliberate opt-in. |
| 17 | `APEX/backend/invest/research/edgar.py:187` | `sec.gov`, `data.sec.gov` (`:89-91`) |
| 18 | `APEX/backend/invest/research/calendar.py:125` | `federalreserve.gov`, `api.stlouisfed.org`, **`api.nasdaq.com` per-symbol** (`:85-87`, `:510`) |
| 19 | `APEX/backend/invest/research/policy.py:722` | `federalregister.gov` (`:404`), `whitehouse.gov` feeds (`:451-452`), `ustr.gov` (`:465`) |
| 20 | `APEX/backend/invest/signals/ingest/chatter.py:334` | ⚠️ `reddit.com` RSS (`:445`), `hn.algolia.com` (`:519`), `api.stocktwits.com` (`:669`) — **queried by ticker**, with a self-identifying `User-Agent` naming the GitHub repo (`:87`). The most fingerprintable of these calls. |
| 21 | `APEX/backend/invest/signals/ingest/gdelt.py:235` | `data.gdeltproject.org` (**plain HTTP**, `:167`), `api.gdeltproject.org` (`:168`) |
| 22 | `APEX/backend/invest/signals/ingest/expectations.py:154` | `clevelandfed.org` (`:270-272`), `philadelphiafed.org` (`:514`), `atlantafed.org` (`:672`) |
| 23 | `APEX/backend/invest/signals/ingest/vintages.py:830` | `api.stlouisfed.org/fred/series/observations` (`:218`) |
| 24 | `APEX/backend/invest/signals/ingest/feeds.py` | `federalreserve.gov` FOMC pages (`:407-409`, `:677`) |
| 25 | `APEX/backend/invest/backtest/backfill.py:30`, `:57` | **Yahoo Finance** via `yf.Ticker(...).history(...)` — unauthenticated, sends tickers |
| 26 | `APEX/scripts/*` | Operator-run collectors: `kalshi_resolved_pull.py:119`, `measure_weather_efficiency.py:146`, `validate_crypto_threshold.py:181`, `collect_temp_data.py:105,175`, `collect_lowtemp_data.py:106,179`, `import_tick_history.py:351`, `kalshi_smoke.py:111`, plus the `backfill_*` / `validate_*` family. Never invoked by the daemon. |

### 4.4 Model-weight download

| # | file:line | Destination | Gate |
| --- | --- | --- | --- |
| 27 | `APEX/backend/rag/embeddings.py:24` | `huggingface.co` — `SentenceTransformer(model_name)` downloads `all-MiniLM-L6-v2` on first use (`config.py:35`) | **Unreachable by default.** `APEX/backend/rag/factory.py:26-28` returns the `knowledge` backend unless `APEX_RAG_BACKEND=qdrant`. `factory.py:8-10` flags the download explicitly. |

### 4.5 The one real gap: `settings.ollama_host` has no loopback guard

`OllamaClient.__init__` (`APEX/backend/llm/client.py:156`) accepts any host:

```python
self.host = (host or settings.ollama_host).rstrip("/")
```

It is then posted to directly (`client.py:202`) with **no validation that it is loopback**. Since
`OllamaClient` is what the router's PERSONAL branch calls (`router.py:74-76`), setting
`OLLAMA_HOST=http://some-box:11434` would send the user's portfolio, cash and transactions
off-machine with every structural gate in the system reporting green — because the router's
guarantee is "PERSONAL goes to `self.local`," and it has no way to know that `local` is no longer
local.

By contrast, `APEX/scripts/hearth_offload_probe.py:213` refuses a non-loopback base URL and
fail-closes, with `_is_loopback` at `:257-262`. **That guard belongs on `OllamaClient` and on the
new `HearthClient`.** It is ~6 lines and closes the only place where APEX's privacy model rests on
configuration rather than code.

### 4.6 What is confirmed absent

Grep across `backend/`, `cli/`, `app/`, `ios/`, `mcp-server/`, `container/` for
`sentry|posthog|segment|analytics|telemetry|amplitude|mixpanel|datadog|opentelemetry|icloud|dropbox|s3.amazonaws|gcs|firebase|supabase`:

- **No third-party telemetry, analytics, or crash reporting.** The only `telemetry` hits are
  APEX's own local token accounting (`APEX/backend/llm/budget.py`).
- **No cloud sync.** The only `icloud` hit is a comment in `APEX/backend/security/encryption.py:514`
  warning *against* leaving plaintext where iCloud would back it up.
- **No remote log sinks.** Every `logger.add` targets `sys.stderr`
  (`APEX/backend/invest/cli.py:1428`, `backtest/scenarios.py:31`, `backtest/run.py:99`).
- **The SwiftUI/iOS app talks only to localhost** — `APEX/app/APEX/Sources/Config.swift:42`
  builds `http://{host}:{port}/api`, and the FastAPI surface rejects any non-loopback client at
  `APEX/backend/api/app.py:43-46` in addition to binding `127.0.0.1`
  (`APEX/backend/serve.py:64`, `:79`).
- **`APEX/FINANCES/` is not read by any parser.** It holds `.numbers`, `.md` and a `dossiers/`
  directory; `APEX/backend/processors/csv_processor.py:23` handles `.csv`/`.xlsx`/`.xls` only —
  there is no `.numbers` reader in the codebase. Filenames only; no contents were opened.

### 4.7 Egress verdict

APEX's privacy posture is **strong, and stronger than its own docs claim**. The finance half is
genuinely airtight — it is not merely policy-gated, it contains no network code. The trading half
necessarily talks to a broker and to public data sources, and that is inherent to what it does.

Three things to fix, in priority order:

1. **Add the loopback guard to `OllamaClient`** (§4.5). One config typo currently defeats the
   entire PERSONAL boundary.
2. **Route `LLMAnalystStrategy` through `LLMRouter`** (§1.5). It is the one personal-data prompt
   that bypasses the gate, held safe only by its caller.
3. **Document the APNs payload** (`apns.py:97-115`). Anyone enabling `APEX_APNS_*` should know they
   are sending symbol/side/dollar-amount/rationale to Apple. It is a reasonable trade for Face ID
   approval — but it should be a decision, not a surprise.

---

## 5. What APEX already does well that HEARTH must not rebuild

### 5.1 HEARTH's ingestion gap is total

This is worth stating plainly, because it is easy to assume HEARTH's RAG can read documents.
**It cannot read anything but plain UTF-8 text.**

`RagIndex.ingest` (`HEARTH/src/hearth/memory/rag.py:107`) walks a directory
(`_walk_text_files`, `rag.py:186`) and calls `_read_text` (`rag.py:197`), which sniffs the first
4096 bytes, returns `None` on any NUL byte, and otherwise does `path.read_text(encoding="utf-8")`.

Consequences:

- **No extension dispatch, no MIME sniffing, no format registry.** There is no `if suffix ==
  ".pdf"` anywhere in the repo.
- **PDF / DOCX / XLSX / PPTX are silently dropped** — they carry NUL bytes in their headers, so
  `_read_text` returns `None` and `ingest` just `continue`s (`rag.py:117-119`). No warning, no
  counter, no error. Point `hearth rag ingest` at a folder of statements and it reports
  `0 files, 0 chunks` and **exits 0**.
- **CSV "works" only by accident** — it is UTF-8, so it gets chunked as raw characters. No
  delimiter parsing, no header preservation, no row-boundary awareness. A chunk can split
  mid-row and loses the header entirely.
- **No OCR, no images, no HTML→text, no JSON structure, no `.ipynb`, no email.**
- A UTF-16 or latin-1 file raises `UnicodeDecodeError` internally and is silently skipped
  (`rag.py:205`).

A grep of `src/`, `pyproject.toml`, `config/*.yaml` and `docs/*.md` for
`pdf|docx|xlsx|csv|ocr|tesseract|pypdf|openpyxl|pandas|unstructured|markitdown|pptx` returns
**zero hits**. No parsing dependency exists in any extra.

> **Live signal.** `HEARTH/src/hearth/config.py:62-71` has uncommitted additions —
> `file_roots` (a colon-separated directory allowlist, deny-by-default) and
> `file_max_bytes` — with a comment describing them as guards for *"the path-taking MCP tools."*
> **Nothing consumes them yet**; the tools they guard do not exist in `mcp/server.py` or
> `mcp/tools.py`. Meanwhile `config/models.yaml` just gained a 14B entry annotated *"financial
> synthesis, categorization, narrative"* and a 3B entry annotated *"small fine-tuning target …
> categorize, extract."*
>
> Read together, someone is scaffolding agent-facing document ingestion right now, plausibly for
> financial documents. **This section is the argument for not writing the parsers.**

### 5.2 The APEX modules HEARTH should consume, not reimplement

| Module | Why it cannot be re-derived cheaply |
| --- | --- |
| **`APEX/backend/processors/base.py`** — *the crown jewel* | The accumulated knowledge of how real bank exports actually look. `DATE_ALIASES` / `DESCRIPTION_ALIASES` / `AMOUNT_ALIASES` / `DEBIT_ALIASES` / `CREDIT_ALIASES` / `TYPE_ALIASES` / `ACCOUNT_ALIASES` / `BALANCE_ALIASES` at `:90-107`, resolved through a normalized index (lowercased, non-alphanumerics stripped) with **alias priority preserved** so the right column wins when a file has several (`_first`, `:77`). `_coerce_money` (`:24-51`) handles parenthesized accounting negatives `(1,234.56)`, trailing-minus `1234.56-`, `$` and comma stripping, and routes floats through `str()` so `0.1` stays `0.1` rather than its binary expansion. `normalize_transaction` (`:269`) and `detect_document_type` (`:243`) sit on top. **This is empirical, not derivable** — every alias in that list is a bank that formatted its export differently, and the only way to obtain it is to have hit them. |
| **`APEX/backend/processors/csv_processor.py`** | CSV + XLSX + XLS via pandas (`:23`), with `detect_format` (`:105`) inferring the layout before extraction. |
| **`APEX/backend/processors/pdf_processor.py`** | **Dual extraction** — pdfplumber tables *and* PyPDF2 text, with results merged (`:70`, `:73`, `:77`), because bank statements put transactions in either. Degrades gracefully when either library is missing (`:9-20`) rather than failing the import. |
| **`APEX/backend/processors/image_processor.py`** | pytesseract OCR with OpenCV/PIL preprocessing (`:5-15`), reusing the PDF text-extraction logic for the OCR'd output (`:32`) so a photographed statement takes the same path as a digital one. |
| **`APEX/backend/processors/manager.py`** | The `can_process` → dispatch registry (`:22-27`, `:33-41`) — the extension point HEARTH's `_read_text` lacks entirely. |
| **`APEX/backend/rag/knowledge_rag.py:36-38`** | Transaction→text chunk rendering batched at `_TXN_BATCH = 10`, *"one chunk ≈ one statement page's worth of line items."* Domain-tuned chunking that generic character chunking cannot reproduce. |

### 5.3 The recommended division of labour

**APEX owns bytes→text. HEARTH owns text→vectors→answers.**

That line is clean, it matches each repo's competence, and it needs exactly **one small change on
the HEARTH side**:

> **Add a text-payload ingest.** `POST /v1/hearth/rag/ingest`
> (`HEARTH/src/hearth/gateway/app.py:239`) currently takes `paths: list[str]` — server-side
> filesystem paths. Add an alternative body shape accepting `{collection, text, source,
> metadata}`, mapping onto a new `RagIndex.ingest_text(...)` beside `ingest(...)`
> (`HEARTH/src/hearth/memory/rag.py:107`). The chunking, `_chunk_id` dedup (`rag.py:229`),
> embedding and storage all already exist and are unchanged — this is a new entry point onto the
> same pipeline, perhaps 40 lines.
>
> This also avoids a real privacy problem: without it, APEX would have to write text extracted
> from a bank PDF to a temp file so HEARTH could read the path — creating a new place for
> unencrypted financial data to land on disk, in a repo whose `backend/security/encryption.py`
> exists precisely to prevent that.

Two further HEARTH fixes worth doing regardless of APEX:

1. **Stop silently dropping binaries.** `rag.py:117-119` should at minimum count and report
   skipped files, so `hearth rag ingest ./statements` cannot report success on zero work. The
   current behaviour is indistinguishable from an empty directory.
2. **Do not add PDF/OCR parsing to HEARTH.** If a HEARTH-side ingest is ever needed, depend on
   or vendor APEX's `processors/` package rather than writing a second one. A second bank-CSV
   alias table will drift from the first, and the drift will be silent — the wrong column parsed
   as an amount produces a plausible number, not an error.

---

## 6. Summary of recommended actions

**APEX — privacy hardening (do these regardless of integration):**

1. Add a loopback guard to `OllamaClient.__init__` (`APEX/backend/llm/client.py:156`), copying
   `_is_loopback` from `APEX/scripts/hearth_offload_probe.py:257-262`. — §4.5
2. Route `LLMAnalystStrategy.propose` through `LLMRouter` with `DataClass.PERSONAL`
   (`APEX/backend/invest/strategies/llm_analyst.py:85`). — §1.5
3. Document the APNs payload contents in `docs/PRIVACY.md`
   (`APEX/backend/invest/authority/apns.py:97-115`). — §4.2
4. Delete the dead `OLLAMA_BASE_URL` from `.env.example:5`. — §1.2

**APEX — HEARTH integration (~1–2 days):**

5. Add `backend/llm/hearth_client.py` (`HearthClient` + `FallbackClient` + loopback guard),
   config knobs, a `select_local_client()` factory, and edit the five construction sites. — §2.3
6. Amend `docs/ui/HEARTH_BRAIN.md` for the four corrections in §2.4.
7. Measure the research council's refutation rate before and after — it is the one metric the
   JSON-mode loss will move. — §2.5

**HEARTH — to make itself consumable here:**

8. Add a text-payload variant to `POST /v1/hearth/rag/ingest` + `RagIndex.ingest_text`. — §5.3
9. Report skipped files from `RagIndex.ingest` instead of silently dropping them
   (`src/hearth/memory/rag.py:117-119`). — §5.3
10. **Do not build PDF/CSV/XLSX/OCR parsing.** APEX's `backend/processors/` already exists and
    encodes empirical knowledge that a fresh implementation will not have. — §5.2
11. Consider `finish_reason: "length"` when `max_tokens` truncates
    (`src/hearth/gateway/schemas.py:56` currently hardcodes `"stop"`) — silent truncation is hard
    to debug from the client side. — §2.2c
