# Time Series Data Compression Framework

This library provides an extensible framework for time series data compression algorithms. It aims to reduce storage requirements while maintaining the ability to perform analytics on the compressed data.

## Features

- Extensible framework for implementing various compression algorithms
- Multiple compression algorithms, both lossless and lossy
- Easy-to-use interface for compressing and decompressing time series data
- Support for numpy arrays

## Implemented Algorithms

1. **Difference Encoding (Lossless)**: A simple compression technique that stores the differences between consecutive values instead of the actual values.

2. **Piecewise Aggregate Approximation (PAA) (Lossy)**: Reduces the time series from n dimensions to w dimensions by dividing the data into w equal-sized frames and calculating the mean values for each frame.

3. **Symbolic Aggregate approXimation (SAX) (Lossy)**: Extends PAA by further discretizing the PAA representation into a small alphabet of symbols, providing an even more compact representation of the time series.

4. **Discrete Cosine Transform (DCT) (Lossy)**: Applies the DCT to the time series and keeps only a specified number of coefficients, effectively compressing the data by discarding high-frequency components.

5. **Run Length Encoding (RLE) (Lossless)**: Compresses data by replacing consecutive data elements with a single data value and count.

6. **Zlib Compression (Lossless)**: Uses the zlib library to compress the time series data, which is particularly effective for data with repeating patterns.

7. **Discrete Wavelet Transform (DWT) (Lossy)**: Applies the DWT to the time series and compresses the data by thresholding the wavelet coefficients, effectively removing small details while preserving the overall structure of the data.

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