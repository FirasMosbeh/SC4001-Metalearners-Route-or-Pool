"""CLIP image features for every image in the manifest, saved with data.features.

Same setup as Ojha et al. (CVPR 2023): frozen ViT-L/14, 768-d projected image embedding,
test-time preprocessing (center crop 224, CLIP mean/std), no augmentation.
Only images without features yet are computed, so a stopped run continues where it left off
and images added to the manifest later are picked up by running it again.
"""
import time

import torch
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor

from common.device import get_device
from common.log import Progress, format_duration, log
from data.features import FeatureStore
from data.images import image_loader, load_manifest

MODEL_NAME = "openai/clip-vit-large-patch14"
FEATURE_NAME = "clip_vit-l14"
BATCH_SIZE = 32  # fastest batch sizes were 8-64 on an M2 Pro; 32 leaves memory headroom
NUM_WORKERS = 4
LOG_EVERY = 50  # batches

INFO = {
    "model": MODEL_NAME,
    "output": "get_image_features().pooler_output (projected, not normalized)",
    "preprocess": "CenterCrop(224), ToTensor, Normalize(CLIP mean/std)",
}


def load_model(device):
    """Returns (model, preprocess)."""
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    image_processor = CLIPProcessor.from_pretrained(MODEL_NAME).image_processor
    # Like Ojha et al. at test time: center crop 224 from our 256 images, CLIP mean/std.
    # (CLIPProcessor would instead resize the short side to 224 first, shrinking the whole image.)
    preprocess = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
    ])
    return model, preprocess


def extract(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, limit=None):
    """limit: stop after this many images (for test runs)."""
    store = FeatureStore(FEATURE_NAME, INFO)
    manifest = load_manifest()
    done = store.done_files()
    todo = [file for file in manifest.file if file not in done]
    log(f"{FEATURE_NAME}: {len(manifest)} images in manifest, {len(manifest) - len(todo)} done, {len(todo)} to do")
    if limit is not None:
        todo = todo[:limit]
    if not todo:
        return

    device = get_device()
    model, preprocess = load_model(device)
    log(f"{MODEL_NAME} on {device}, batch size {batch_size}, {num_workers} workers")

    start = time.time()
    progress = Progress(len(todo))
    with torch.no_grad():
        for batch, (pixels, indices) in enumerate(image_loader(todo, preprocess, batch_size, num_workers), 1):
            features = model.get_image_features(pixel_values=pixels.to(device)).pooler_output
            store.add([todo[i] for i in indices.tolist()], features.cpu().numpy())
            progress.step(len(indices))
            if batch % LOG_EVERY == 0:
                finish = time.strftime("%H:%M", time.localtime(time.time() + progress.remaining_seconds()))
                log(f"{progress}, done at ~{finish}")

    store.flush()
    log(f"{FEATURE_NAME}: {len(todo)} images in {format_duration(time.time() - start)}")


if __name__ == "__main__":
    extract()
