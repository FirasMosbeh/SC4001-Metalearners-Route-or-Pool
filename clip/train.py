"""Train one logistic regression per expert (real vs. that expert's fakes) on the saved CLIP features.

Same model as notebooks/clip/probe_one_generator.ipynb: StandardScaler + LogisticRegression, trained on
the expert's expert_train images and evaluated on its expert_val images. Every expert uses the same
images as its ResNet counterpart (resnet.experts.expert_rows). meta_train, meta_val and test are never
touched here.

Finished experts are saved and skipped on a re-run.
"""
import time

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from clip.extract import FEATURE_NAME
from common.log import log
from data import config
from data.features import FeatureStore
from data.images import load_manifest
from resnet.experts import EXPERTS, check_no_leaks, expert_rows
from resnet.model import MODELS_DIR

MAX_ITER = 2000


def experts_dir():
    """Folder with one model per expert, next to the ResNet experts."""
    return config.dataset_dir() / MODELS_DIR / FEATURE_NAME


def model_path(expert):
    return experts_dir() / f"{expert}.joblib"


def load_expert(expert):
    """Returns (fitted pipeline, info)."""
    saved = joblib.load(model_path(expert))
    return saved["model"], saved["info"]


def train_expert(expert, manifest, store):
    """Train, evaluate and save one expert. Returns its info. Skipped if the model already exists."""
    path = model_path(expert)
    if path.exists():
        log(f"[{expert}] already trained, skipping (delete {path} to retrain)")
        return load_expert(expert)[1]

    train_rows = expert_rows(manifest, expert, "expert_train")
    val_rows = expert_rows(manifest, expert, "expert_val")
    X_train, y_train = store.load(train_rows.file), train_rows.label.to_numpy()
    X_val, y_val = store.load(val_rows.file), val_rows.label.to_numpy()
    log(f"[{expert}] {len(train_rows)} train images ({y_train.mean():.0%} fake), "
        f"{len(val_rows)} val images, generators {EXPERTS[expert]}")

    start = time.time()
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=MAX_ITER))
    model.fit(X_train, y_train)
    p_fake = model.predict_proba(X_val)[:, 1]
    info = {
        "expert": expert,
        "generators": EXPERTS[expert],
        "features": FEATURE_NAME,
        "val_acc": float(accuracy_score(y_val, p_fake > 0.5)),
        "val_auc": float(roc_auc_score(y_val, p_fake)),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "train_seconds": round(time.time() - start, 1),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary file first, so a crash mid-write can't leave a broken model that gets skipped.
    temporary = path.with_suffix(".tmp")
    joblib.dump({"model": model, "info": info}, temporary)
    temporary.replace(path)
    log(f"[{expert}] saved: val acc {info['val_acc']:.4f}, auc {info['val_auc']:.4f}")
    return info


def train_all(experts=None):
    """Train every expert that isn't trained yet. Returns {expert: info}."""
    manifest = load_manifest()
    check_no_leaks(manifest)
    store = FeatureStore(FEATURE_NAME)
    return {expert: train_expert(expert, manifest, store) for expert in experts or EXPERTS}


if __name__ == "__main__":
    train_all()
