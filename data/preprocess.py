import io

from PIL import Image

from data.config import CROP_SIZE, JPEG_QUALITY


def preprocess(raw):
    """Decode, center-crop to CROP_SIZE (bicubic upscale first if smaller), re-encode as JPEG.

    No downscaling: large images are only cropped. Returns (jpeg_bytes, orig_format, orig_w, orig_h).
    """
    img = Image.open(io.BytesIO(raw))
    orig_format, (w, h) = img.format, img.size
    img = img.convert("RGB")

    if min(w, h) < CROP_SIZE:
        scale = CROP_SIZE / min(w, h)
        new_size = (max(CROP_SIZE, round(w * scale)), max(CROP_SIZE, round(h * scale)))
        img = img.resize(new_size, Image.BICUBIC)

    left = (img.width - CROP_SIZE) // 2
    top = (img.height - CROP_SIZE) // 2
    img = img.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue(), orig_format, w, h
