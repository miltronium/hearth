"""The default HEARTH_EMBED_MODEL must be an id the registry actually registers (B-073).

It used to be ``mlx-community/bge-small-en-v1.5-mlx``, which config/models.yaml itself
records as 404ing upstream, so ``HEARTH_EMBEDDER=mlx`` with the default could never resolve.
(Separately, B-011: the MLX embedder cannot load BERT weights yet.)
"""

from __future__ import annotations

from hearth.config import Settings
from hearth.registry import load_registry


def test_the_default_embed_model_is_a_registered_embedding_model():
    default = Settings.model_fields["embed_model"].default
    entries = {e.id: e for e in load_registry().list()}
    assert default in entries, f"{default!r} is not in config/models.yaml"
    assert "embed" in entries[default].capabilities
