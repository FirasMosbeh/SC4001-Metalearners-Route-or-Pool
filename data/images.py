"""Reading the downloaded images: the manifest, filtering it, and loading images for a model."""
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data import config

LABEL_IDS = {name: label for label, name in config.LABEL_NAMES.items()}  # "real" -> 0, "fake" -> 1


def load_manifest():
    """All downloaded images, one row each: file, label, generator, pool, ..."""
    return pd.read_csv(config.dataset_dir() / config.MANIFEST_FILE)


def _as_list(value):
    return value if isinstance(value, (list, tuple, set)) else [value]


def select(manifest, pool=None, generator=None, label=None):
    """Rows matching all given filters. Each filter takes one value or a list; None means no filter.

    pool:      expert_train, expert_val, meta_train, meta_val, test
    generator: one of config.GENERATORS
    label:     0 / 1 or "real" / "fake"
    """
    keep = pd.Series(True, index=manifest.index)
    if pool is not None:
        keep &= manifest.pool.isin(_as_list(pool))
    if generator is not None:
        keep &= manifest.generator.isin(_as_list(generator))
    if label is not None:
        labels = [LABEL_IDS.get(value, value) for value in _as_list(label)]
        keep &= manifest.label.isin(labels)
    return manifest[keep]


def open_image(file):
    """file: path relative to the dataset folder, as in the manifest's "file" column."""
    return Image.open(config.dataset_dir() / file).convert("RGB")


class ImageDataset(Dataset):
    """Images as model input: returns (transform(image), index into files).

    Defined here and not in a notebook: DataLoader workers on macOS start as new processes
    and can't load classes defined in a notebook.
    """

    def __init__(self, files, transform):
        self.files = list(files)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return self.transform(open_image(self.files[index])), index


def image_loader(files, transform, batch_size, num_workers):
    """Batches of (images, indices) in the order of files."""
    return DataLoader(ImageDataset(files, transform), batch_size=batch_size,
                      shuffle=False, num_workers=num_workers)
