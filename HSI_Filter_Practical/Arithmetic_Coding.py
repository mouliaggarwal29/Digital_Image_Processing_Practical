
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Read Input Image
ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "input" / "input.jpg"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

img = cv2.imread(str(INPUT_PATH), cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Input image not found!")

pixels = img.flatten()

# 2. Calculate Symbol Probabilities
symbols, counts = np.unique(
    pixels, return_counts=True
)

probabilities = counts / len(pixels)

# Create cumulative probability intervals
cumulative = np.concatenate((
    [0.0], np.cumsum(probabilities)
))

# Map pixel values to their intervals
intervals = {}

for i, symbol in enumerate(symbols):
    intervals[int(symbol)] = (
        cumulative[i],
        cumulative[i + 1]
    )

# 3. Arithmetic Encoding
def arithmetic_encode(data, intervals):
    low = 0.0
    high = 1.0

    for symbol in data:
        range_value = high - low

        symbol_low, symbol_high = intervals[int(symbol)]

        high = low + range_value * symbol_high
        low = low + range_value * symbol_low

    return (low + high) / 2

encoded_value = arithmetic_encode(
    pixels, intervals
)

# 4. Calculate Compression Statistics
# A fixed 64-bit representation is used to store
# the arithmetic code value in this demonstration.
original_bits = img.size * 8
compressed_bits = 64

compression_ratio = original_bits / compressed_bits
compression_percentage = (
    (1 - compressed_bits / original_bits) * 100
)

print("\n--- ARITHMETIC CODING RESULTS ---")
print("Original Image Size:", original_bits, "bits")
print("Arithmetic Code Value:", encoded_value)
print("Stored Code Size:", compressed_bits, "bits")
print("Theoretical Demonstration Ratio:",
      round(compression_ratio, 2))
print("Theoretical Space Saving:",
      round(compression_percentage, 2), "%")

# 5. Arithmetic Decoding
def arithmetic_decode(value, total_pixels, symbols,
                      cumulative):

    decoded = []

    for _ in range(total_pixels):
        # Find the interval containing the code value
        index = np.searchsorted(
            cumulative, value, side='right'
        ) - 1

        index = min(index, len(symbols) - 1)

        decoded.append(symbols[index])

        # Update code value to the selected interval
        low = cumulative[index]
        high = cumulative[index + 1]

        value = (value - low) / (high - low)

    return np.array(decoded, dtype=np.uint8)

decoded_pixels = arithmetic_decode(
    encoded_value,
    len(pixels),
    symbols,
    cumulative
)

reconstructed = decoded_pixels.reshape(img.shape)

# 6. Save Results
with open(OUTPUT_DIR / "arithmetic_encoded.txt", "w") as f:
    f.write(str(encoded_value))

cv2.imwrite(
    str(OUTPUT_DIR / "arithmetic_reconstructed.png"),
    reconstructed
)

# 7. Display Results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(reconstructed, cmap='gray')
plt.title("Arithmetic Reconstructed Image")
plt.axis("off")

plt.suptitle("Arithmetic Image Compression")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "arithmetic_comparison.png",
    dpi=150
)
plt.show()

print("\nArithmetic Decoding Successful!")
print("Images saved in output folder.")