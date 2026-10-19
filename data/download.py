import json
import math
import time

import numpy as np

from common.coarse_to_fine import coarse_to_fine
from common.log import Progress, log
from data import config
from data.dataset_writer import DatasetWriter
from data.shard_log import ShardLog
from data.shards import list_repo_files, list_shards, open_shards

TRAIN_POOLS = ["expert_train", "expert_val", "meta_train", "meta_val"]


def seeded_rng(generator, *keys):
    """Random number generator seeded by SEED, the generator and extra keys, so every draw is reproducible."""
    return np.random.default_rng([config.SEED, config.GENERATORS.index(generator), *keys])


def pick_pool(remaining, rng):
    """Random pool, weighted by how many images each pool still needs."""
    pools = [pool for pool, count in remaining.items() if count > 0]
    counts = np.array([remaining[pool] for pool in pools], dtype=float)
    return pools[rng.choice(len(pools), p=counts / counts.sum())]


def save_images(candidates, remaining, generator, label, writer, progress, rng, limit=None):
    """Offer candidate images to the writer, in order, until `limit` are saved or all pools are full.

    candidates: (table, row) pairs of images with this label
    remaining:  pool -> images this label still needs there; decremented for every saved image
    """
    saved = 0
    for table, row in candidates:
        if not any(remaining.values()) or (limit is not None and saved >= limit):
            break
        pool = pick_pool(remaining, rng)
        source_path = table["image_path"][row].as_py()
        image_bytes = table["image"][row].as_py()
        if writer.add(pool, generator, label, source_path, image_bytes):
            remaining[pool] -= 1
            saved += 1
            progress.step()


def still_needed(remaining):
    return f"still need real={sum(remaining[0].values())} fake={sum(remaining[1].values())}"


def take_from_shard(generator, shard_files, shard_index, caps, remaining, writer, progress):
    """Download one train shard, save up to caps[label] new images per label, delete it.

    Returns the set of labels the shard contains.
    """
    description = f"[train] {generator} shard {shard_index}/{len(shard_files)}"
    with open_shards([shard_files[shard_index]], description) as (table,):
        labels = table.column("label").to_numpy()
        for label in config.LABEL_NAMES:
            rows = np.flatnonzero(labels == label)
            # Same seed on every visit, so a revisit continues down the same random order.
            seeded_rng(generator, label, shard_index).shuffle(rows)
            candidates = ((table, row) for row in rows)
            save_images(candidates, remaining[label], generator, label, writer, progress,
                        rng=seeded_rng(generator, label, shard_index, 1), limit=caps[label])
        writer.flush()
    log(f"  {still_needed(remaining)} | {progress}")
    return {int(label) for label in np.unique(labels)}


def sample_train(generator, writer, shard_log, progress, repo_files):
    """Fill the train-derived pools for one generator, shard by shard."""
    sizes = config.pool_sizes()
    # label -> pool -> images still needed
    remaining = {
        label: {pool: sizes[pool] - writer.counts[(pool, generator, label)] for pool in TRAIN_POOLS}
        for label in config.LABEL_NAMES
    }
    shard_files = list_shards(repo_files, "train", generator)
    # shard index -> labels it contains, learned while downloading (including earlier runs)
    labels_in_shard = {
        index: set(shard_log.labels[shard_file])
        for index, shard_file in enumerate(shard_files) if shard_file in shard_log.labels
    }

    def done():
        return not any(any(pools.values()) for pools in remaining.values())

    # Pass 1: coarse-to-fine over all shards, at most MAX_PER_LABEL_PER_SHARD per label per shard.
    caps = {label: config.MAX_PER_LABEL_PER_SHARD for label in config.LABEL_NAMES}
    for shard_index in coarse_to_fine(len(shard_files)):
        if done():
            break
        if shard_index in labels_in_shard:
            continue  # already visited by an earlier run
        labels_in_shard[shard_index] = take_from_shard(
            generator, shard_files, shard_index, caps, remaining, writer, progress)
        shard_log.record(shard_files[shard_index], labels_in_shard[shard_index])

    # Pass 2 (top-up): if a label is still short, e.g. because it sits in only a few
    # shards (BigGAN fakes), revisit only the shards known to contain it and spread
    # the missing amount evenly across them.
    short_labels = [label for label in config.LABEL_NAMES if any(remaining[label].values())]
    if short_labels:
        shards_with_label = {
            label: [index for index, labels in labels_in_shard.items() if label in labels]
            for label in short_labels
        }
        caps = {label: 0 for label in config.LABEL_NAMES}
        for label in short_labels:
            if shards_with_label[label]:
                caps[label] = math.ceil(sum(remaining[label].values()) / len(shards_with_label[label]))
        revisit = sorted({index for label in short_labels for index in shards_with_label[label]})
        log(f"[train] {generator}: top-up, {still_needed(remaining)}, "
            f"revisiting {len(revisit)} shards with cap real={caps[0]} fake={caps[1]}")
        for shard_index in revisit:
            if done():
                break
            take_from_shard(generator, shard_files, shard_index, caps, remaining, writer, progress)

    if not done():
        log(f"WARNING {generator}: ran out of shards, {still_needed(remaining)}")


def sample_test(generator, writer, progress, repo_files):
    """Draw the fixed test subsample uniformly from the generator's whole official test set."""
    target = config.pool_sizes()["test"]
    remaining = {
        label: {"test": target - writer.counts[("test", generator, label)]}
        for label in config.LABEL_NAMES
    }
    if not any(remaining[label]["test"] for label in config.LABEL_NAMES):
        return

    with open_shards(list_shards(repo_files, "test", generator), f"[test] {generator}") as tables:
        for label in config.LABEL_NAMES:
            candidates = [
                (table, row)
                for table in tables
                for row in np.flatnonzero(table.column("label").to_numpy() == label)
            ]
            # Same random order on every run, so a resumed run ends with the same test set.
            order = seeded_rng(generator, label, 999_999).permutation(len(candidates))
            save_images((candidates[index] for index in order), remaining[label], generator, label,
                        writer, progress, rng=seeded_rng(generator, label, 999_999))
        writer.flush()
    log(f"  {progress}")


def download_all():
    """Download, sample and preprocess every pool for every generator. Safe to re-run (resumes)."""
    dataset_dir = config.dataset_dir()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "params.json").write_text(json.dumps(config.hashed_params(), indent=2))
    log(f"Writing to {dataset_dir}")

    writer = DatasetWriter(dataset_dir)
    sizes = config.pool_sizes()
    total_needed = sum(
        max(0, sizes[pool] - writer.counts[(pool, generator, label)])
        for pool in sizes for generator in config.GENERATORS for label in config.LABEL_NAMES
    )
    progress = Progress(total_needed)
    log(f"{sum(writer.counts.values())} images already saved, {total_needed} still needed")

    shard_log = ShardLog(dataset_dir)
    repo_files = list_repo_files()
    try:
        for generator in config.GENERATORS:
            log(f"=== {generator} ===")
            sample_train(generator, writer, shard_log, progress, repo_files)
            sample_test(generator, writer, progress, repo_files)
    finally:
        # Also on a crash: writes out buffered manifest rows, so a restart sees every saved image.
        writer.close()
    log(f"Done. {progress}")


def download_all_with_restarts(attempts=5, wait_seconds=60):
    """download_all(), restarted after a crash. Each restart resumes where the last one stopped."""
    for attempt in range(1, attempts + 1):
        try:
            download_all()
            return
        except Exception as error:  # KeyboardInterrupt is not an Exception, so stopping by hand still works
            if attempt == attempts:
                raise
            log(f"ERROR {error!r} (attempt {attempt}/{attempts}), restarting in {wait_seconds}s")
            time.sleep(wait_seconds)
