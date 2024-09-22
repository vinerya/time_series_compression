import numpy as np
import matplotlib.pyplot as plt
from time_series_compression import TimeSeriesCompressor, DifferenceEncoding, PAA, SAX, DCT, RunLengthEncoding, ZlibCompression, DiscreteWaveletTransform

# Generate sample time series data
time = np.arange(0, 10, 0.01)
data = np.sin(time) + np.random.normal(0, 0.1, time.shape)

# Create a TimeSeriesCompressor instance
compressor = TimeSeriesCompressor()

# List of algorithms to test
algorithms = [
    ("DifferenceEncoding (Lossless)", DifferenceEncoding()),
    ("PAA (Lossy)", PAA(segments=50)),
    ("SAX (Lossy)", SAX(segments=50, alphabet_size=10)),
    ("DCT (Lossy)", DCT(keep_coeffs=50)),
    ("RunLengthEncoding (Lossless)", RunLengthEncoding()),
    ("ZlibCompression (Lossless)", ZlibCompression()),
    ("DiscreteWaveletTransform (Lossy)", DiscreteWaveletTransform(wavelet='db4', level=5, threshold=0.1))
]

# Compress, decompress, and evaluate each algorithm
results = []
plt.figure(figsize=(15, 10))
plt.plot(time, data, label='Original Data', alpha=0.5)

for name, algo in algorithms:
    compressor.set_algorithm(algo)
    
    # Compress and decompress
    compressed_data = compressor.compress(data)
    decompressed_data = compressor.decompress(compressed_data)
    
    # Calculate compression ratio and MSE
    if isinstance(compressed_data, list):
        if all(isinstance(item, tuple) for item in compressed_data):  # For RunLengthEncoding
            compressed_size = sum(len(str(item)) for item in compressed_data)
        else:  # For DWT
            compressed_size = sum(arr.nbytes if isinstance(arr, np.ndarray) else len(str(arr)) for arr in compressed_data)
    elif isinstance(compressed_data, bytes):  # For ZlibCompression
        compressed_size = len(compressed_data)
    else:
        compressed_size = compressed_data.nbytes
    
    compression_ratio = data.nbytes / compressed_size
    
    # Check if shapes match before calculating MSE
    if data.shape == decompressed_data.shape:
        mse = np.mean((data - decompressed_data) ** 2)
    else:
        print(f"Warning: Shape mismatch for {name}. Original: {data.shape}, Decompressed: {decompressed_data.shape}")
        mse = np.nan
    
    results.append((name, compression_ratio, mse))
    
    # Plot decompressed data
    plt.plot(time[:len(decompressed_data)], decompressed_data, label=f'Decompressed ({name})')

plt.legend()
plt.title('Original vs Decompressed Data')
plt.xlabel('Time')
plt.ylabel('Value')
plt.savefig('compression_result.png')
plt.close()

print("Plot saved as 'compression_result.png'")

# Print results
print("\nCompression Algorithm Results:")
print("--------------------------------")
for name, ratio, mse in results:
    print(f"{name}:")
    print(f"  Compression Ratio: {ratio:.2f}")
    print(f"  Mean Squared Error: {mse:.6f}" if not np.isnan(mse) else "  Mean Squared Error: N/A (shape mismatch)")
    print("--------------------------------")