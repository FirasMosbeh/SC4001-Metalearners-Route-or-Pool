"""Finding, downloading and reading GenImage shards on the Hugging Face Hub."""
import multiprocessing
import os
import time
from contextlib import contextmanager

import pyarrow as pa
from huggingface_hub import HfApi, hf_hub_download

from common.log import log
from data import config

# None falls back to the token saved by `hf auth login`, if there is one.
TOKEN = os.environ.get("HF_TOKEN")


def list_repo_files():
    return HfApi().list_repo_files(config.HF_REPO, repo_type="dataset", token=TOKEN)


def list_shards(repo_files, split, generator):
    """Sorted shard files of one generator in one split ("train" or "test").

    The repo stores them as data/<split>/<generator>/data-00000-of-00075.arrow, ... next to
    a few small JSON files. This keeps only the .arrow files of that folder.
    """
    folder = f"data/{split}/{generator}/"
    return sorted(f for f in repo_files if f.startswith(folder) and f.endswith(".arrow"))


def _download(shard_file, scratch_dir):
    hf_hub_download(
        config.HF_REPO, shard_file, repo_type="dataset",
        local_dir=scratch_dir, token=TOKEN,
    )


def download_shard(shard_file):
    """Download one shard into SCRATCH_DIR and return its local path.

    The download runs in a separate process, so a hung download (which raises no error)
    can be killed after DOWNLOAD_TIMEOUT_SECONDS. Failed or hung downloads are retried
    up to DOWNLOAD_ATTEMPTS times; partial downloads resume where they stopped.
    """
    path = config.SCRATCH_DIR / shard_file
    for attempt in range(1, config.DOWNLOAD_ATTEMPTS + 1):
        process = multiprocessing.get_context("spawn").Process(
            target=_download, args=(shard_file, config.SCRATCH_DIR))
        process.start()
        process.join(config.DOWNLOAD_TIMEOUT_SECONDS)
        if process.is_alive():
            process.kill()
            process.join()
            problem = f"timed out after {config.DOWNLOAD_TIMEOUT_SECONDS}s"
        elif process.exitcode == 0 and path.exists():
            return str(path)
        else:
            problem = f"failed (exit code {process.exitcode})"

        if attempt < config.DOWNLOAD_ATTEMPTS:
            wait = config.RETRY_WAIT_SECONDS * attempt
            log(f"WARNING download of {shard_file} {problem} "
                f"(attempt {attempt}/{config.DOWNLOAD_ATTEMPTS}), retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"download of {shard_file} {problem} {config.DOWNLOAD_ATTEMPTS} times, giving up")


def read_shard(path):
    """Open a shard as an Arrow table with columns image_path, image, label, generator, ...

    Memory-mapped, so only the rows we actually touch are read from disk.
    """
    return pa.ipc.open_stream(pa.memory_map(str(path))).read_all()


@contextmanager
def open_shards(shard_files, description):
    """Download shards, yield them as tables, delete them afterwards. Logs size and timings.

    If the block fails, the shards stay in SCRATCH_DIR, so a re-run doesn't download them again.
    """
    start = time.time()
    paths = [download_shard(shard_file) for shard_file in shard_files]
    size_gb = sum(os.path.getsize(path) for path in paths) / 1e9
    log(f"{description}: downloaded {size_gb:.2f} GB in {time.time() - start:.0f}s")

    start = time.time()
    yield [read_shard(path) for path in paths]
    for path in paths:
        os.remove(path)
    log(f"{description}: processed in {time.time() - start:.0f}s")
