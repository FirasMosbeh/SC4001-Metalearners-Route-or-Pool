"""Scores of every CLIP expert on every image the combiners use, saved with data.features.

Counterpart of resnet/score.py: one feature set "clip_vit-l14_experts" with one column per expert holding
its raw logit (> 0 = fake). Feature columns aren't named, so the experts and the scored pools come from
resnet (EXPERTS, score_pools) and the columns line up with the ResNet scores. Only images without
scores yet are computed.
"""
import hashlib

import numpy as np

from clip.extract import FEATURE_NAME
from clip.train import load_expert, model_path
from common.log import log
from data.features import FeatureStore
from data.images import load_manifest, select
from resnet.experts import EXPERTS
from resnet.score import score_pools

SCORES_NAME = f"{FEATURE_NAME}_experts"


def store_info():
    """What produced the scores. Retrained experts have other file hashes, so their scores can't end up
    in the same store as the old ones (FeatureStore raises)."""
    return {
        "model": "StandardScaler + LogisticRegression on " + FEATURE_NAME,
        "output": "raw logit per expert (decision_function, > 0 = fake), one column per expert in the order of 'experts'",
        "experts": EXPERTS,
        "checkpoints": {expert: hashlib.md5(model_path(expert).read_bytes()).hexdigest()[:12] for expert in EXPERTS},
    }


def score():
    """Score every image of the scored pools that has no scores yet."""
    store = FeatureStore(SCORES_NAME, store_info())
    manifest = load_manifest()
    rows = select(manifest, pool=score_pools(manifest))
    done = store.done_files()
    todo = [file for file in rows.file if file not in done]
    log(f"{SCORES_NAME}: {len(rows)} images in pools {score_pools(manifest)}, "
        f"{len(rows) - len(todo)} done, {len(todo)} to do")
    if not todo:
        return

    X = FeatureStore(FEATURE_NAME).load(todo)
    logits = np.stack([load_expert(expert)[0].decision_function(X) for expert in EXPERTS], axis=1)
    store.add(todo, logits)
    store.flush()
    log(f"{SCORES_NAME}: scored {len(todo)} images with {len(EXPERTS)} experts")


if __name__ == "__main__":
    score()
