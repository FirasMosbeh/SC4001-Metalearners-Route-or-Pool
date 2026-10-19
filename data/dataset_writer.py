import csv
import hashlib
import re
from collections import Counter

from data import config
from data.preprocess import preprocess

MANIFEST_COLUMNS = [
    "file", "label", "generator", "pool",
    "orig_path", "orig_format", "orig_w", "orig_h", "md5", "class_token",
]

# ImageNet class ID (a WordNet "synset"), e.g. n01440764 in "ADM/train/nature/n01440764_10026.JPEG".
# Only real train images have it in their filename.
IMAGENET_CLASS_ID = re.compile(r"n\d{8}")


class DatasetWriter:
    """Saves processed images and their manifest rows into the dataset folder.

    On start it reads the existing manifest, so it knows what is already saved (for resuming)
    and never takes the same image twice, checked by source path and by md5 of the image bytes.
    """

    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        manifest_path = dataset_dir / config.MANIFEST_FILE
        is_new = not manifest_path.exists()

        rows = []
        if not is_new:
            with open(manifest_path) as f:
                rows = list(csv.DictReader(f))
        self.used_paths = {row["orig_path"] for row in rows}
        self.used_md5s = {row["md5"] for row in rows}
        # (pool, generator, label) -> number of images saved
        self.counts = Counter((row["pool"], row["generator"], int(row["label"])) for row in rows)

        self._file = open(manifest_path, "a", newline="")
        self._csv = csv.DictWriter(self._file, fieldnames=MANIFEST_COLUMNS)
        if is_new:
            self._csv.writeheader()

    def add(self, pool, generator, label, source_path, image_bytes):
        """Preprocess one image, save it and add its manifest row.

        pool:         split the image goes to, e.g. "expert_train" or "test"
        generator:    GenImage generator, e.g. "ADM"
        label:        0 = real, 1 = fake
        source_path:  path of the image inside GenImage (the `image_path` column),
                      e.g. "ADM/train/ai/428_adm_50.PNG"
        image_bytes:  the original image file (PNG or JPEG bytes from the `image` column)

        Returns False if the image was already used or can't be decoded.
        """
        if source_path in self.used_paths:
            return False
        md5 = hashlib.md5(image_bytes).hexdigest()
        if md5 in self.used_md5s:
            return False
        try:
            jpeg_bytes, orig_format, orig_w, orig_h = preprocess(image_bytes)
        except OSError:
            return False

        file = f"{pool}/{generator}/{config.LABEL_NAMES[label]}/{md5}.jpg"
        (self.dataset_dir / file).parent.mkdir(parents=True, exist_ok=True)
        (self.dataset_dir / file).write_bytes(jpeg_bytes)

        class_id = IMAGENET_CLASS_ID.search(source_path)
        self._csv.writerow({
            "file": file, "label": label, "generator": generator, "pool": pool,
            "orig_path": source_path, "orig_format": orig_format, "orig_w": orig_w, "orig_h": orig_h,
            "md5": md5, "class_token": class_id.group(0) if class_id else "",
        })
        self.used_paths.add(source_path)
        self.used_md5s.add(md5)
        self.counts[(pool, generator, label)] += 1
        return True

    def flush(self):
        self._file.flush()

    def close(self):
        self._file.close()
