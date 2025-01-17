# Enhanced Time Series Data Compression Framework

This library provides a comprehensive framework for time series data compression algorithms, featuring advanced capabilities for both batch and streaming data processing. It aims to reduce storage requirements while maintaining analytical capabilities and offering performance optimization features.

## Features

- Extensible framework for implementing various compression algorithms
- Multiple compression algorithms, both lossless and lossy
- Easy-to-use interface for compressing and decompressing time series data
- Support for numpy arrays
- Parallel processing capabilities for large datasets
- Streaming data support for real-time compression
- Automatic algorithm selection based on data characteristics
- Comprehensive benchmarking suite
- Type hints and extensive documentation

## Core Algorithms

### Lossless Algorithms

1. **Difference Encoding**: Stores differences between consecutive values
2. **Run Length Encoding (RLE)**: Compresses consecutive data elements into value-count pairs
3. **Zlib Compression**: Uses zlib library for general-purpose compression
4. **Delta-RLE Hybrid**: Combines delta encoding with RLE for efficient compression of data with constant changes

### Lossy Algorithms

1. **Piecewise Aggregate Approximation (PAA)**: Reduces dimensions by segment averaging
2. **Symbolic Aggregate approXimation (SAX)**: Extends PAA with symbol discretization
3. **Discrete Cosine Transform (DCT)**: Preserves significant frequency components
4. **Discrete Wavelet Transform (DWT)**: Uses wavelets for multi-resolution compression
5. **PCA Compression**: Reduces dimensionality while preserving data variance

## Advanced Features

### Parallel Processing
```python
# Compress large datasets in parallel
compressed_chunks = compressor.compress_parallel(data, chunk_size=1000)
decompressed_data = compressor.decompress_parallel(compressed_chunks)
```

### Streaming Data Support
```python
# Process streaming data
stream_compressor = StreamingCompressionAlgorithm()
for chunk in data_stream:
    compressed_chunk = stream_compressor.partial_compress(chunk)
final_compressed = stream_compressor.finalize_compression()
```

### Automatic Algorithm Selection
```python
# Let the framework choose the best algorithm
best_algo = compressor.auto_select_algorithm(
    data, 
    algorithms,
    priority='balanced'  # Options: 'size', 'speed', 'accuracy'
)
```

### Benchmarking
```python
# Compare algorithm performance
results = compressor.benchmark_all(data, algorithms)
print(results)  # Shows compression ratio, MSE, processing times
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/time_series_compression.git
   ```

2. Navigate to the project directory:
   ```
   cd time_series_compression
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Install the library:
   ```
   pip install .
   ```

## Usage

Here's a simple example of how to use the Time Series Compression Framework:

```python
import numpy as np
from time_series_compression import TimeSeriesCompressor, DifferenceEncoding, PAA, SAX, DCT, RunLengthEncoding, ZlibCompression, DiscreteWaveletTransform

# Generate sample time series data
time = np.arange(0, 10, 0.1)
data = np.sin(time) + np.random.normal(0, 0.1, time.shape)

# Create a TimeSeriesCompressor instance
compressor = TimeSeriesCompressor()

# Compress and decompress using different algorithms
algorithms = [
    ("DifferenceEncoding (Lossless)", DifferenceEncoding()),
    ("PAA (Lossy)", PAA(segments=10)),
    ("SAX (Lossy)", SAX(segments=10, alphabet_size=5)),
    ("DCT (Lossy)", DCT(keep_coeffs=10)),
    ("RunLengthEncoding (Lossless)", RunLengthEncoding()),
    ("ZlibCompression (Lossless)", ZlibCompression()),
    ("DiscreteWaveletTransform (Lossy)", DiscreteWaveletTransform(wavelet='db4', level=3, threshold=0.1))
]

for name, algo in algorithms:
    compressor.set_algorithm(algo)
    compressed_data = compressor.compress(data)
    decompressed_data = compressor.decompress(compressed_data)
    
    print(f"{name}:")
    print(f"  Original data shape: {data.shape}")
    print(f"  Compressed data shape: {compressed_data.shape if hasattr(compressed_data, 'shape') else len(compressed_data)}")
    print(f"  Decompressed data shape: {decompressed_data.shape}")
```

## Extending the Framework

You can create your own compression algorithms by subclassing the `CompressionAlgorithm` abstract base class:

```python
class CustomAlgorithm(CompressionAlgorithm):
    def compress(self, data):
        # Implement compression logic
        return compressed_data

    def decompress(self, compressed_data):
        # Implement decompression logic
        return decompressed_data

# Use the new algorithm
compressor = TimeSeriesCompressor(CustomAlgorithm())
compressed_data = compressor.compress(data)
decompressed_data = compressor.decompress(compressed_data)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
