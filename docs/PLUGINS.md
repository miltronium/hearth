# HEARTH — Plugins

**Status:** Phase 7. Companion to [ARCHITECTURE.md](ARCHITECTURE.md) (§4 providers, §5
registry, §6 memory) and [DECISIONS.md](DECISIONS.md) (ADR-004 single `ModelProvider`,
ADR-008 `VectorStore`).

HEARTH grows **without core edits**. A third-party package registers a backend through a
Python packaging *entry point*; once installed, HEARTH discovers it and resolves it by name
through the same selection functions the built-in backends use. No fork, no patch, no PR to
HEARTH — this is ADR-004 ("adding a backend = one new class + a registry entry") extended to
external packages.

---

## Entry-point groups

Declare an entry point in one of these three groups. The **value** is a `module:attr` target
that resolves to a zero-argument **factory** returning an instance (a plain class whose
`__init__` takes no required arguments works directly as its own factory).

| Group | Backend kind | Protocol the instance must satisfy | Selected by |
| --- | --- | --- | --- |
| `hearth.providers` | inference backend | [`ModelProvider`](../src/hearth/providers/base.py) | `HEARTH_BACKEND=<name>` |
| `hearth.vector_stores` | RAG vector store | [`VectorStore`](../src/hearth/memory/store.py) | `HEARTH_VECTOR_STORE=<name>` |
| `hearth.embedders` | embedding backend | [`EmbeddingProvider`](../src/hearth/memory/embed.py) | `HEARTH_EMBEDDER=<name>` |

The Protocols are `runtime_checkable`, so a plugin satisfies one **structurally** — it needs
the right attributes and methods, not a HEARTH base class. A plugin does not even need to
depend on `hearth`.

### The Protocols (what your instance must provide)

`ModelProvider` (see `src/hearth/providers/base.py`):

```python
name: str
def capabilities(self) -> Capabilities: ...
def generate(self, req: GenRequest) -> GenResult: ...
def stream(self, req: GenRequest) -> Iterator[str]: ...
def footprint(self, model_id: str) -> ResourceEstimate: ...   # ram_gb drives scheduling
```

`VectorStore` (see `src/hearth/memory/store.py`):

```python
def add(self, collection: str, chunks: list[Chunk], vectors: list[list[float]]) -> int: ...
def query(self, collection: str, vector: list[float], k: int) -> list[Chunk]: ...
def count(self, collection: str) -> int: ...
```

`EmbeddingProvider` (see `src/hearth/memory/embed.py`):

```python
name: str
dim: int
def embed(self, texts: list[str]) -> list[list[float]]: ...
```

`footprint().ram_gb` matters: the multi-model [`ModelManager`](../src/hearth/serving/manager.py)
uses it to keep resident models within `HEARTH_RAM_CEILING_GB`, LRU-evicting when a new load
would overflow. Report an honest estimate.

---

## Writing a plugin

A minimal, runnable reference lives in [`examples/plugin/`](../examples/plugin/). It is a
standalone package (own `pyproject.toml`) with a trivial provider and this entry-point table:

```toml
[project.entry-points."hearth.providers"]
hello = "hearth_hello:build"
```

`hearth_hello:build` is a factory returning a `HelloProvider()`. That table is the entire
integration.

### Try it

```console
$ cd examples/plugin && pip install -e .          # or: uv pip install -e .
$ hearth plugins list
  name    group             target               status
  hello   hearth.providers  hearth_hello:build   ok
$ HEARTH_BACKEND=hello hearth run "hi there"
  Hello from the example plugin! You said: hi there
```

No HEARTH source changed. (The example is a reference/fixture — HEARTH does not install it.)

---

## Discovery, validation, and failure handling

`hearth plugins list` shows every entry point in the three groups with its status:

- **ok** — imported, the factory ran, and the instance satisfies the group's Protocol.
- **skipped — <reason>** — import error, factory raised, or the instance failed the
  Protocol check.

Failure is **per-plugin and graceful** (see `src/hearth/plugins.py`): a broken plugin logs a
warning and is skipped — it never crashes discovery, `select_provider`, or `hearth serve`.
When `HEARTH_BACKEND=<name>` names a plugin that fails to load, selection degrades and raises
a clear error rather than serving a half-constructed backend.

---

## Publishing

1. Package your backend normally (`pyproject.toml` / `setup.cfg`), depending only on what your
   backend needs — a dependency on `hearth` is optional since the interface is a Protocol.
2. Declare the entry point in the appropriate group (table above).
3. Publish to your index (internal PyPI mirror or a git/path install).
4. Install it alongside HEARTH; set the matching `HEARTH_*` env var to your entry-point name.

Version your plugin against the Protocol shapes documented here. These are stable contracts
(ADR-004, ADR-008); if HEARTH ever changes one, it will be a deliberate, documented break.
