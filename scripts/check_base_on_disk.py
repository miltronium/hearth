"""Is a base model on disk? The pre-check ``scripts/train_lora_real.sh`` runs (B-024).

Prints the local path and exits 0 when ``model_id`` resolves from disk; prints the
resolver's error (which carries the ``hearth models pull`` hint) and exits 3 when it does
not. Uses ``hearth.providers.mlx.resolve_local_model`` — the resolver every load path,
including ``hearth train`` itself, uses — so the pre-check and the training run agree on
where models live: HEARTH's ``~/.hearth/models`` (where ``hearth models pull`` writes)
first, then the huggingface hub cache. Downloads are always off here.

    python scripts/check_base_on_disk.py <model-id>
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: check_base_on_disk.py <model-id>", file=sys.stderr)
        return 2
    from hearth.providers.mlx import ModelNotOnDiskError, resolve_local_model

    try:
        path = resolve_local_model(args[0], allow_downloads=False)
    except ModelNotOnDiskError as exc:
        print(f"NOT ON DISK: {exc}", file=sys.stderr)
        return 3
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
