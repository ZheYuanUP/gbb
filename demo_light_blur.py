"""Preview the lighting and Gaussian-blur augmentations used during training."""

import random
from pathlib import Path

from PIL import Image, ImageDraw
from torchvision.transforms.functional import adjust_brightness, gaussian_blur


# ==================== Input and output ====================
# Change this variable to the image you want to inspect.
INPUT_IMAGE = "blur_light/craw_far_1_1.png"
OUTPUT_DIR = "blur_light/craw_far_1_1_changed.png"

# ==================== Augmentation settings ====================
# These defaults match scripts/train_hedian.sh.
RANDOM_SEED = 1
LIGHT_BRIGHTNESS_MIN = 0.8
LIGHT_BRIGHTNESS_MAX = 1.2
GAUSSIAN_BLUR_KERNEL_SIZE = 5
GAUSSIAN_BLUR_SIGMA_MIN = 0.1
GAUSSIAN_BLUR_SIGMA_MAX = 2.0


def add_label(image, label, label_height=30):
    """Add a simple label above an image for the comparison canvas."""
    canvas = Image.new("RGB", (image.width, image.height + label_height), "white")
    canvas.paste(image, (0, label_height))
    ImageDraw.Draw(canvas).text((8, 8), label, fill="black")
    return canvas


def horizontal_comparison(items):
    """Build one horizontal image from (label, image) pairs."""
    panels = [add_label(image, label) for label, image in items]
    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    comparison = Image.new("RGB", (width, height), "white")
    offset = 0
    for panel in panels:
        comparison.paste(panel, (offset, 0))
        offset += panel.width
    return comparison


def main():
    input_path = Path(INPUT_IMAGE)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {input_path.resolve()}")

    if GAUSSIAN_BLUR_KERNEL_SIZE <= 0 or GAUSSIAN_BLUR_KERNEL_SIZE % 2 == 0:
        raise ValueError("GAUSSIAN_BLUR_KERNEL_SIZE must be a positive odd integer.")

    rng = random.Random(RANDOM_SEED)
    brightness = rng.uniform(LIGHT_BRIGHTNESS_MIN, LIGHT_BRIGHTNESS_MAX)
    sigma = rng.uniform(GAUSSIAN_BLUR_SIGMA_MIN, GAUSSIAN_BLUR_SIGMA_MAX)
    kernel_size = [GAUSSIAN_BLUR_KERNEL_SIZE, GAUSSIAN_BLUR_KERNEL_SIZE]
    blur_sigma = [sigma, sigma]

    with Image.open(input_path) as source:
        original = source.convert("RGB")

    light_only = adjust_brightness(original, brightness)
    blur_only = gaussian_blur(original, kernel_size, blur_sigma)
    light_and_blur = gaussian_blur(light_only, kernel_size, blur_sigma)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "01_original.png": original,
        "02_light_only.png": light_only,
        "03_gaussian_blur_only.png": blur_only,
        "04_light_and_blur.png": light_and_blur,
    }
    for filename, image in outputs.items():
        image.save(output_dir / filename)

    comparison = horizontal_comparison(
        [
            ("Original", original),
            (f"Light: {brightness:.3f}", light_only),
            (f"Blur sigma: {sigma:.3f}", blur_only),
            ("Light + blur", light_and_blur),
        ]
    )
    comparison.save(output_dir / "00_comparison.png")

    print(f"Input: {input_path.resolve()}")
    print(f"Brightness factor: {brightness:.4f}")
    print(f"Gaussian sigma: {sigma:.4f}")
    print(f"Results: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
