"""The 7 experts of a pool and the images each one trains on.

GenImage has 8 generators, but SD 1.4 and SD 1.5 are near-identical, so they share one expert:
7 experts per pool. The CLIP experts should use the same EXPERTS, so both pools line up.
"""
import numpy as np

from data import config
from data.images import select

# expert -> generators it is trained on
EXPERTS = {
    "ADM": ["ADM"],
    "BigGAN": ["BigGAN"],
    "glide": ["glide"],
    "Midjourney": ["Midjourney"],
    "SD": ["stable_diffusion_v_1_4", "stable_diffusion_v_1_5"],
    "VQDM": ["VQDM"],
    "wukong": ["wukong"],
}


def expert_rows(manifest, expert, pool, limit=None):
    """Rows of one pool the expert trains on: its generators' reals and fakes, sorted by file.

    Every expert gets the same number of images per label as a single generator has in the pool,
    so the merged SD expert takes half of SD 1.4 and half of SD 1.5 instead of twice as many images.
    limit: at most this many images per label (for test runs).
    """
    generators = EXPERTS[expert]
    per_label = config.pool_sizes()[pool] if limit is None else limit
    per_generator = -(-per_label // len(generators))  # rounded up
    parts = []
    for generator in generators:
        for label in config.LABEL_NAMES:
            rows = select(manifest, pool=pool, generator=generator, label=label).sort_values("file")
            rng = np.random.default_rng([config.SEED, config.GENERATORS.index(generator), label])
            keep = np.sort(rng.permutation(len(rows))[:per_generator])
            parts.append(rows.iloc[keep])
    rows = np.concatenate([part.index for part in parts])
    return manifest.loc[rows].sort_values("file")


def check_no_leaks(manifest):
    """Raise if an image could be in two pools: same source image or same bytes under two manifest rows."""
    for column in ("file", "orig_path", "md5"):
        duplicates = manifest[manifest[column].duplicated(keep=False)]
        if len(duplicates):
            raise ValueError(f"{len(duplicates)} manifest rows share a {column}, "
                             f"e.g. {duplicates[column].iloc[0]} in pools {sorted(set(duplicates.pool))}")
