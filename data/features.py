"""Saving and loading features computed from the images (CLIP, ResNet, ...), keyed by image file.

Each feature set has its own folder dataset_dir/features/<name>/ with
    info.json        what produced the features (model, preprocessing, ...)
    part_0000.npz    files (n,), features (n, d)
    part_0001.npz    ...
Every part holds exactly PART_SIZE images, except the last one of a run (the rest). Parts are found
by the files they contain, not by position. So an extraction can stop at any point, and images added
to the manifest later only need their own new parts.
"""
import json

import numpy as np
import pandas as pd

from common.log import log
from data import config
from data.images import load_manifest, select

FEATURES_DIR = "features"  # inside the dataset folder
INFO_FILE = "info.json"
PART_SIZE = 5000  # images per part; a crash during extraction loses at most one unfinished part


class FeatureStore:
    def __init__(self, name, info=None):
        """info: what produced the features, saved on first use. If the store already has a different
        info.json, this raises, so features of another model can't end up under the same name.
        Leave it out to only read.
        """
        self.dir = config.dataset_dir() / FEATURES_DIR / name
        self._buffer_files, self._buffer_features = [], []
        info_path = self.dir / INFO_FILE
        if info is None:
            return
        if info_path.exists():
            saved = json.loads(info_path.read_text())
            if saved != info:
                raise ValueError(f"{info_path} was made with {saved}, not {info}")
        else:
            self.dir.mkdir(parents=True, exist_ok=True)
            info_path.write_text(json.dumps(info, indent=1))

    def parts(self):
        return sorted(self.dir.glob("part_*.npz"))

    def done_files(self):
        """Files that already have features."""
        done = set()
        for part in self.parts():
            with np.load(part) as data:
                done.update(data["files"])
        return done

    def _write_part(self, files, features):
        """Save one part with the next free number. Returns its path."""
        numbers = [int(part.stem.split("_")[1]) for part in self.parts()]
        path = self.dir / f"part_{max(numbers, default=-1) + 1:04d}.npz"
        # Write to a temporary file first, so a crash mid-write can't leave a broken part.
        temporary = path.with_suffix(".tmp")
        with open(temporary, "wb") as f:
            np.savez(f, files=np.asarray(files, dtype=str), features=np.asarray(features, dtype=np.float32))
        temporary.replace(path)
        return path

    def add(self, files, features):
        """Collect features of any number of images; saves a part of exactly PART_SIZE images
        whenever enough are collected. The rest waits for the next add() or flush()."""
        self._buffer_files.extend(files)
        self._buffer_features.append(np.asarray(features, dtype=np.float32))
        while len(self._buffer_files) >= PART_SIZE:
            self._save(PART_SIZE)

    def flush(self):
        """Save the rest collected by add() (fewer than PART_SIZE images). Call once at the end of a run."""
        if self._buffer_files:
            self._save(len(self._buffer_files))

    def _save(self, count):
        """Save the first count collected images as a part and keep the rest."""
        features = np.concatenate(self._buffer_features)
        path = self._write_part(self._buffer_files[:count], features[:count])
        log(f"saved {count} images to {path.name}")
        self._buffer_files = self._buffer_files[count:]
        self._buffer_features = [features[count:]]

    def load(self, files):
        """Features of files, one row per file in the same order. Raises if any file has none."""
        all_files, all_features = [], []
        for part in self.parts():
            with np.load(part) as data:
                all_files.append(data["files"])
                all_features.append(data["features"])
        if not all_files:
            raise FileNotFoundError(f"no features in {self.dir}")
        rows = pd.Index(np.concatenate(all_files)).get_indexer(list(files))
        missing = int((rows == -1).sum())
        if missing:
            raise KeyError(f"{missing} of {len(rows)} files have no features in {self.dir}")
        return np.concatenate(all_features)[rows]

    def check(self):
        """Log whether every image in the manifest has features. Returns the manifest rows without features."""
        manifest = load_manifest()
        stored = []
        for part in self.parts():
            with np.load(part) as data:
                stored.extend(data["files"])
        stored_set = set(stored)

        missing = manifest[~manifest.file.isin(stored_set)]
        extra = stored_set - set(manifest.file)
        duplicates = len(stored) - len(stored_set)

        log(f"{self.dir.name}: {len(manifest) - len(missing)}/{len(manifest)} manifest images have features, "
            f"{len(self.parts())} parts")
        if len(missing):
            log(f"MISSING {len(missing)} images:")
            counts = missing.groupby(["pool", "generator", "label"]).size()
            for (pool, generator, label), count in counts.items():
                log(f"  {pool:12s} {generator:24s} {config.LABEL_NAMES[label]:4s} {count}")
        if extra:
            log(f"WARNING {len(extra)} images have features but aren't in the manifest")
        if duplicates:
            log(f"WARNING {duplicates} images have features in more than one part")
        if not len(missing) and not extra and not duplicates:
            log("all good")
        return missing


def load_features(name, pool=None, generator=None, label=None):
    """Features of the manifest rows matching the filters (see data.images.select).

    Returns (X, rows): X has one row of features per row of rows, e.g. y = rows.label.to_numpy().
    """
    rows = select(load_manifest(), pool=pool, generator=generator, label=label)
    return FeatureStore(name).load(rows.file), rows
