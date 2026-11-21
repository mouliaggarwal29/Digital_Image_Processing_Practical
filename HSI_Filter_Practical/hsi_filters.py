import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import (
    minimum_filter, maximum_filter, median_filter,
    uniform_filter, convolve, gaussian_filter
)

# Resolve folders relative to this script, so it runs from VS Code
ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "input" / "input.jpg"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. Read image and convert BGR -> RGB
img_bgr = cv2.imread(str(INPUT_PATH))
if img_bgr is None:
    raise FileNotFoundError(
        f"Could not read {INPUT_PATH}. Put your colour image at input/input.jpg"
    )
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


# 2. RGB to HSI conversion
def rgb_to_hsi(image):
    r, g, b = image[..., 0], image[..., 1], image[..., 2]

    numerator = 0.5 * ((r - g) + (r - b))
    denominator = np.sqrt((r - g)**2 + (r - b) * (g - b))
    theta = np.arccos(np.clip(numerator / (denominator + 1e-8), -1, 1))

    hue = np.where(b <= g, theta, 2 * np.pi - theta) / (2 * np.pi)
    minimum = np.minimum(np.minimum(r, g), b)
    saturation = 1 - (3 * minimum / (r + g + b + 1e-8))
    intensity = (r + g + b) / 3

    black = (r + g + b) == 0
    hue = np.where(black, 0, hue)
    saturation = np.where(black, 0, saturation)

    return np.stack((hue, np.clip(saturation, 0, 1),
                     np.clip(intensity, 0, 1)), axis=-1)


# 3. HSI to RGB conversion
def hsi_to_rgb(hsi):
    h = hsi[..., 0] * 2 * np.pi
    s = np.clip(hsi[..., 1], 0, 1)
    i = np.clip(hsi[..., 2], 0, 1)

    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)

    # Sector 1: 0 <= H < 120 degrees
    mask = h < 2 * np.pi / 3
    hm = h[mask]
    b[mask] = i[mask] * (1 - s[mask])
    r[mask] = i[mask] * (
        1 + s[mask] * np.cos(hm) /
        (np.cos(np.pi / 3 - hm) + 1e-8)
    )
    g[mask] = 3 * i[mask] - r[mask] - b[mask]

    # Sector 2: 120 <= H < 240 degrees
    mask = (h >= 2 * np.pi / 3) & (h < 4 * np.pi / 3)
    hm = h[mask] - 2 * np.pi / 3
    r[mask] = i[mask] * (1 - s[mask])
    g[mask] = i[mask] * (
        1 + s[mask] * np.cos(hm) /
        (np.cos(np.pi / 3 - hm) + 1e-8)
    )
    b[mask] = 3 * i[mask] - r[mask] - g[mask]

    # Sector 3: 240 <= H < 360 degrees
    mask = h >= 4 * np.pi / 3
    hm = h[mask] - 4 * np.pi / 3
    g[mask] = i[mask] * (1 - s[mask])
    b[mask] = i[mask] * (
        1 + s[mask] * np.cos(hm) /
        (np.cos(np.pi / 3 - hm) + 1e-8)
    )
    r[mask] = 3 * i[mask] - g[mask] - b[mask]

    return np.clip(np.stack((r, g, b), axis=-1), 0, 1)


# 4. Apply a spatial filter separately to H, S and I channels
def apply_filter(hsi_image, filter_function):
    output = np.empty_like(hsi_image)
    for channel in range(3):
        output[..., channel] = filter_function(hsi_image[..., channel])
    return output


hsi = rgb_to_hsi(img)

# 5. Apply six filters using a 3x3 neighbourhood
filtered_hsi = {
    "Minimum": apply_filter(hsi, lambda x: minimum_filter(x, size=3)),
    "Median": apply_filter(hsi, lambda x: median_filter(x, size=3)),
    "Maximum": apply_filter(hsi, lambda x: maximum_filter(x, size=3)),
    "Averaging": apply_filter(hsi, lambda x: uniform_filter(x, size=3)),
}

weighted_kernel = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=np.float32) / 16.0

filtered_hsi["Weighted_Average"] = apply_filter(
    hsi, lambda x: convolve(x, weighted_kernel, mode="reflect")
)
filtered_hsi["Gaussian"] = apply_filter(
    hsi, lambda x: gaussian_filter(x, sigma=1.0, mode="reflect")
)

# 6. Convert filtered images back to RGB and save individual results
rgb_results = {"Original": img}
for name, filtered in filtered_hsi.items():
    rgb_results[name] = hsi_to_rgb(filtered)

for name, result in rgb_results.items():
    out = (np.clip(result, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(
        str(OUTPUT_DIR / f"{name.lower()}.png"),
        cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    )

# 7. Create and save a comparison figure
titles = [
    "Original Image", "Minimum Filter", "Median Filter",
    "Maximum Filter", "Averaging Filter",
    "Weighted Average Filter", "Gaussian Filter"
]
keys = [
    "Original", "Minimum", "Median", "Maximum",
    "Averaging", "Weighted_Average", "Gaussian"
]

fig, axes = plt.subplots(2, 4, figsize=(16, 9))
for ax, key, title in zip(axes.flat, keys, titles):
    ax.imshow(rgb_results[key])
    ax.set_title(title)
    ax.axis("off")

axes.flat[-1].axis("off")
fig.suptitle("Spatial Filtering in HSI Colour Space", fontsize=16)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "all_filters_comparison.png", dpi=150,
            bbox_inches="tight")
plt.show()

print("Done! All filtered images are saved in the 'output' folder.")
