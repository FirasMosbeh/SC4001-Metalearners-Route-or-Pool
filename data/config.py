# Data config. All counts are images PER LABEL PER GENERATOR.
import hashlib
import json
import os
from pathlib import Path

HF_REPO = "nebula/GenImage-arrow"

GENERATORS = [
    "ADM",
    "BigGAN",
    "glide",
    "Midjourney",
    "stable_diffusion_v_1_4",
    "stable_diffusion_v_1_5",
    "VQDM",
    "wukong",
]

PILOT = False

LABEL_NAMES = {0: "real", 1: "fake"}

# Full scale
EXPERT_TRAIN = 5000
EXPERT_VAL = 1000
META_TRAIN = 2000
META_VAL = 1000
TEST = 2000

# Pilot (~1/5 scale)
PILOT_EXPERT_TRAIN = 1000
PILOT_EXPERT_VAL = 200
PILOT_META_TRAIN = 400
PILOT_META_VAL = 200
PILOT_TEST = 500

# Sampling
SEED = 4001
# Spreads samples over many shards (rows are sorted by class).
MAX_PER_LABEL_PER_SHARD = 150

# Downloads (a shard is ~0.5 GB, normally ~20s)
# A download taking longer is treated as hung and restarted.
DOWNLOAD_TIMEOUT_SECONDS = 300
DOWNLOAD_ATTEMPTS = 5
# Multiplied by the attempt number.
RETRY_WAIT_SECONDS = 30

# Image size and compression
CROP_SIZE = 256
JPEG_QUALITY = 95

# Folders
INPUT_DIR = Path(__file__).resolve().parent.parent / "input"  # repo root, also when run from notebooks/
SCRATCH_DIR = Path(os.environ.get("SCRATCH_DIR", "/tmp/genimage_scratch"))  # raw shards, deleted after use
MANIFEST_FILE = "manifest.csv"  # inside the dataset folder


def pool_sizes():
    if PILOT:
        return {
            "expert_train": PILOT_EXPERT_TRAIN,
            "expert_val": PILOT_EXPERT_VAL,
            "meta_train": PILOT_META_TRAIN,
            "meta_val": PILOT_META_VAL,
            "test": PILOT_TEST,
        }
    return {
        "expert_train": EXPERT_TRAIN,
        "expert_val": EXPERT_VAL,
        "meta_train": META_TRAIN,
        "meta_val": META_VAL,
        "test": TEST,
    }


def hashed_params():
    # meta_train is left out on purpose: it can grow without changing the folder.
    sizes = {k: v for k, v in pool_sizes().items() if k != "meta_train"}
    return {
        "generators": GENERATORS,
        "pool_sizes": sizes,
        "seed": SEED,
        "max_per_label_per_shard": MAX_PER_LABEL_PER_SHARD,
        "crop_size": CROP_SIZE,
        "jpeg_quality": JPEG_QUALITY,
    }


def dataset_dir():
    params = json.dumps(hashed_params(), sort_keys=True)
    return INPUT_DIR / f"genimage_{hashlib.sha1(params.encode()).hexdigest()[:8]}"
