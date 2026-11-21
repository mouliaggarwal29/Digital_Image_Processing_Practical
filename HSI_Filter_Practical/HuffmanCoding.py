
import cv2
import numpy as np
import matplotlib.pyplot as plt
import heapq
from collections import Counter
from pathlib import Path

# 1. Read Input Image
ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "input" / "input.jpg"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

img = cv2.imread(str(INPUT_PATH), cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Input image not found!")

# 2. Calculate Frequency of Each Pixel
pixels = img.flatten()
frequency = Counter(pixels)

# 3. Create Huffman Tree
class Node:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(frequency):
    heap = []

    for symbol, freq in frequency.items():
        heapq.heappush(heap, Node(symbol, freq))

    if len(heap) == 1:
        return heap[0]

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)

        parent = Node(None, left.freq + right.freq)
        parent.left = left
        parent.right = right

        heapq.heappush(heap, parent)

    return heap[0]

# 4. Generate Huffman Codes
def generate_codes(root):
    codes = {}

    def traverse(node, code):
        if node.symbol is not None:
            codes[node.symbol] = code if code else "0"
            return

        traverse(node.left, code + "0")
        traverse(node.right, code + "1")

    traverse(root, "")
    return codes

root = build_huffman_tree(frequency)
huffman_codes = generate_codes(root)

# 5. Encode Image
encoded_data = ''.join(
    huffman_codes[pixel] for pixel in pixels
)

# 6. Calculate Compression Statistics
original_bits = img.size * 8
compressed_bits = len(encoded_data)

compression_ratio = original_bits / compressed_bits
compression_percentage = (
    (1 - compressed_bits / original_bits) * 100
)

print("\n--- HUFFMAN CODING RESULTS ---")
print("Original Image Size:", original_bits, "bits")
print("Compressed Data Size:", compressed_bits, "bits")
print("Compression Ratio:", round(compression_ratio, 2))
print("Space Saving:", round(compression_percentage, 2), "%")

# 7. Decode Huffman Data
def decode_huffman(encoded, root, total_pixels):
    decoded = []

    if root.symbol is not None:
        return np.full(total_pixels, root.symbol, dtype=np.uint8)

    current = root

    for bit in encoded:
        if bit == '0':
            current = current.left
        else:
            current = current.right

        if current.symbol is not None:
            decoded.append(current.symbol)
            current = root

            if len(decoded) == total_pixels:
                break

    return np.array(decoded, dtype=np.uint8)

decoded_pixels = decode_huffman(
    encoded_data, root, img.size
)

reconstructed = decoded_pixels.reshape(img.shape)

# 8. Save Compressed Data and Reconstructed Image
with open(OUTPUT_DIR / "huffman_encoded.txt", "w") as f:
    f.write(encoded_data)

cv2.imwrite(
    str(OUTPUT_DIR / "huffman_reconstructed.png"),
    reconstructed
)

# 9. Display Results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(reconstructed, cmap='gray')
plt.title("Huffman Reconstructed Image")
plt.axis("off")

plt.suptitle("Huffman Image Compression")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "huffman_comparison.png",
    dpi=150
)
plt.show()

print("\nHuffman Decoding Successful!")
print("Images saved in output folder.")