"""Benchmark candidate local models on this machine (resolves Open Question #1).

Loads each model via the MLX backend, times a few representative coder prompts, and
reports load time, tokens/sec, and peak memory. Requires the mlx extra:

    uv run --extra mlx python scripts/bench.py

Results inform the default_model choice in ROADMAP Phase 0.
"""

from __future__ import annotations

import time

# Candidate models to compare on a 32 GB Apple Silicon machine (4-bit quant).
CANDIDATES = [
    "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit",
]

PROMPTS = [
    "Write a Python function that reverses a linked list.",
    "Summarize what a mutex is in two sentences.",
    "Given this diff, write a conventional-commit message: +def add(a,b): return a+b",
]


def bench_one(model_id: str, max_tokens: int = 128) -> dict:
    from mlx_lm import generate, load

    t0 = time.perf_counter()
    model, tokenizer = load(model_id)
    load_s = time.perf_counter() - t0

    total_tokens = 0
    t1 = time.perf_counter()
    for p in PROMPTS:
        messages = [{"role": "user", "content": p}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        out = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
        total_tokens += len(tokenizer.encode(out))
    gen_s = time.perf_counter() - t1

    return {
        "model": model_id,
        "load_s": round(load_s, 2),
        "gen_s": round(gen_s, 2),
        "tokens": total_tokens,
        "tok_per_s": round(total_tokens / gen_s, 1) if gen_s else 0.0,
    }


def main() -> None:
    for model_id in CANDIDATES:
        try:
            print(bench_one(model_id))
        except Exception as exc:  # noqa: BLE001 — bench should report and continue
            print({"model": model_id, "error": str(exc)})


if __name__ == "__main__":
    main()
