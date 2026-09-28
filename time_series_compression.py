import io
import numpy as np
import pandas as pd
import sys
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Union, List
from scipy.stats import norm
from scipy.fft import dct, idct
from sklearn.decomposition import PCA
import zlib
import pywt
import time

class CompressionAlgorithm(ABC):
    """
    Base class for all compression algorithms.

    An instance holds only its configuration. ``compress`` returns everything
    ``decompress`` needs, so compressed output can be stored or sent to another
    process and decompressed by any instance configured with the same parameters.
    """
    @abstractmethod
    def compress(self, data):
        pass

    @abstractmethod
    def decompress(self, compressed_data):
        pass

class DifferenceEncoding(CompressionAlgorithm):
    def compress(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a numpy array")

        compressed = np.zeros_like(data)
        compressed[0] = data[0]
        compressed[1:] = np.diff(data)
        return compressed

    def decompress(self, compressed_data):
        if not isinstance(compressed_data, np.ndarray):
            raise TypeError("Input data must be a numpy array")

        return np.cumsum(compressed_data)

    def __repr__(self):
        return "DifferenceEncoding()"

class PAA(CompressionAlgorithm):
    def __init__(self, segments):
        if segments <= 0:
            raise ValueError("segments must be a positive integer")
        self.segments = segments

    def compress(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a numpy array")

        # Use array_split to handle non-divisible lengths without data loss;
        # never ask for more segments than there are points
        chunks = np.array_split(data, min(self.segments, len(data)))
        means = np.array([chunk.mean() for chunk in chunks])
        return means, len(data)

    def decompress(self, compressed_data):
        if not isinstance(compressed_data, tuple):
            raise TypeError("Input data must be a (means, original_length) tuple")

        means, original_length = compressed_data
        # Reconstruct with exact original length using same split logic
        chunk_sizes = [len(c) for c in np.array_split(np.empty(original_length), len(means))]
        return np.repeat(means, chunk_sizes)

    def __repr__(self):
        return f"PAA(segments={self.segments})"

class SAX(CompressionAlgorithm):
    def __init__(self, segments, alphabet_size):
        if segments <= 0:
            raise ValueError("segments must be a positive integer")
        if alphabet_size <= 1:
            raise ValueError("alphabet_size must be greater than 1")
        self.segments = segments
        self.alphabet_size = alphabet_size
        self.breakpoints = norm.ppf(np.linspace(0, 1, alphabet_size + 1)[1:-1])
        # Each symbol decodes to the mean of the standard normal within its bin:
        # E[Z | a < Z < b] = (pdf(a) - pdf(b)) / (cdf(b) - cdf(a)), where every
        # bin holds probability 1 / alphabet_size
        edges = np.concatenate(([-np.inf], self.breakpoints, [np.inf]))
        self.centroids = alphabet_size * (norm.pdf(edges[:-1]) - norm.pdf(edges[1:]))

    def compress(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a numpy array")

        mean = np.mean(data)
        std = np.std(data)

        # Normalize the data (constant data has zero spread, so skip scaling)
        normalized_data = (data - mean) / (std if std > 0 else 1.0)

        # PAA compression
        paa_data, _ = PAA(self.segments).compress(normalized_data)

        # Discretize to symbols
        symbolic_data = np.digitize(paa_data, self.breakpoints)
        return symbolic_data, len(data), mean, std

    def decompress(self, compressed_data):
        if not isinstance(compressed_data, tuple):
            raise TypeError("Input data must be a (symbols, original_length, mean, std) tuple")

        symbolic_data, original_length, mean, std = compressed_data

        # Convert symbols back to PAA
        paa_data = self.centroids[symbolic_data]

        # PAA decompression
        decompressed = PAA(self.segments).decompress((paa_data, original_length))

        # Denormalize
        return decompressed * std + mean

    def __repr__(self):
        return f"SAX(segments={self.segments}, alphabet_size={self.alphabet_size})"

class DCT(CompressionAlgorithm):
    def __init__(self, keep_coeffs):
        if keep_coeffs <= 0:
            raise ValueError("keep_coeffs must be a positive integer")
        self.keep_coeffs = keep_coeffs

    def compress(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a numpy array")
        
        dct_coeffs = dct(data)
        return dct_coeffs[:self.keep_coeffs], len(data)

    def decompress(self, compressed_data):
        if not isinstance(compressed_data, tuple):
            raise TypeError("Input data must be a (coefficients, original_length) tuple")

        coeffs, original_length = compressed_data
        full_coeffs = np.zeros(original_length)
        full_coeffs[:len(coeffs)] = coeffs
        return idct(full_coeffs)

    def __repr__(self):
        return f"DCT(keep_coeffs={self.keep_coeffs})"

class RunLengthEncoding(CompressionAlgorithm):
    def compress(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a numpy array")
        
        compressed = []
        count = 1
        for i in range(1, len(data)):
            if data[i] == data[i-1]:
                count += 1
            else:
                compressed.append((data[i-1], count))
                count = 1
        compressed.append((data[-1], count))
        return compressed

    def decompress(self, compressed_data):
        decompressed = []
        for value, count in compressed_data:
            decompressed.extend([value] * count)
        return np.array(decompressed)

    def __repr__(self):
        return "RunLengthEncoding()"

class ZlibCompression(CompressionAlgorithm):
    def compress(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a numpy array")

        # The .npy header records dtype and shape, so the output is self-describing
        buffer = io.BytesIO()
        np.save(buffer, data, allow_pickle=False)
        return zlib.compress(buffer.getvalue())

    def decompress(self, compressed_data):
        if not isinstance(compressed_data, bytes):
            raise TypeError("Input data must be bytes")

        decompressed = zlib.decompress(compressed_data)
        return np.load(io.BytesIO(decompressed), allow_pickle=False)

    def __repr__(self):
        return "ZlibCompression()"

class DiscreteWaveletTransform(CompressionAlgorithm):
    def __init__(self, wavelet='db4', level=None, threshold=0.1):
        self.wavelet = wavelet
        self.level = level
        self.threshold = threshold

    def compress(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a numpy array")
        
        coeffs = pywt.wavedec(data, self.wavelet, level=self.level)

        # Threshold the coefficients
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], self.threshold * np.max(np.abs(coeffs[i])))

        return coeffs, len(data)

    def decompress(self, compressed_data):
        if not isinstance(compressed_data, tuple):
            raise TypeError("Input data must be a (coefficients, original_length) tuple")

        coeffs, original_length = compressed_data
        # waverec can return one extra sample for odd-length input
        return pywt.waverec(coeffs, self.wavelet)[:original_length]

    def __repr__(self):
        return f"DiscreteWaveletTransform(wavelet='{self.wavelet}', level={self.level}, threshold={self.threshold})"

class StreamingCompressionAlgorithm(CompressionAlgorithm, ABC):
    """Base class for streaming compression algorithms."""
    
    @abstractmethod
    def partial_compress(self, chunk: np.ndarray) -> np.ndarray:
        """Compress a chunk of streaming data."""
        pass

    @abstractmethod
    def finalize_compression(self) -> np.ndarray:
        """Finalize the compression after all chunks are processed."""
        pass

class DeltaRLE(CompressionAlgorithm):
    """
    Hybrid compression algorithm combining delta encoding with RLE.
    Particularly effective for time series with long periods of constant change.

    Every reconstructed value stays within ``tolerance`` of the original (up to
    floating-point rounding), so ``tolerance=0`` gives lossless compression.
    """

    def __init__(self, tolerance: float = 1e-6):
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        self.tolerance = tolerance

    def compress(self, data: np.ndarray) -> list:
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a numpy array")
        if len(data) == 0:
            raise ValueError("Input data cannot be empty")
        
        compressed = []

        # Calculate deltas
        deltas = np.diff(data)
        if len(deltas) == 0:
            return [(data[0], 0, 0.0)]

        current_delta = deltas[0]
        count = 1
        start_val = data[0]

        # Extend the run while the value decompress would rebuild stays within
        # tolerance. Comparing against the rebuilt value (rather than comparing
        # deltas) stops small per-step differences from accumulating.
        for i in range(1, len(deltas)):
            predicted = start_val + (count + 1) * current_delta
            if abs(predicted - data[i + 1]) <= self.tolerance:
                count += 1
            else:
                compressed.append((start_val, count, current_delta))
                start_val = data[i]
                current_delta = deltas[i]
                count = 1

        compressed.append((start_val, count, current_delta))
        return compressed

    def __repr__(self):
        return f"DeltaRLE(tolerance={self.tolerance})"

    def decompress(self, compressed_data: list) -> np.ndarray:
        decompressed = []

        for idx, (start_val, count, delta) in enumerate(compressed_data):
            values = [start_val + i * delta for i in range(count + 1)]
            if idx < len(compressed_data) - 1:
                # Drop last value (shared boundary with next segment)
                decompressed.extend(values[:-1])
            else:
                decompressed.extend(values)

        return np.array(decompressed)

class PCACompression(CompressionAlgorithm):
    """
    PCA-based compression algorithm.

    Multivariate data of shape (n_samples, n_features) is reduced across its
    features. A univariate series is first cut into windows of ``window_size``
    points that become the rows of a matrix, so what gets compressed is the
    correlation between neighbouring points. ``window_size=None`` uses the
    square root of the series length, which roughly minimizes the stored size.
    """

    def __init__(self, n_components: Optional[Union[int, float]] = 0.95,
                 window_size: Optional[int] = None):
        if window_size is not None and window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        self.n_components = n_components
        self.window_size = window_size

    def compress(self, data: np.ndarray) -> tuple:
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a numpy array")
        if len(data) == 0:
            raise ValueError("Input data cannot be empty")

        if data.ndim == 1:
            window = self.window_size or max(1, int(np.sqrt(len(data))))
            n_windows = -(-len(data) // window)
            # Pad with the last value so the series fills whole windows
            padded = np.pad(data, (0, n_windows * window - len(data)), mode='edge')
            matrix = padded.reshape(n_windows, window)
        elif data.ndim == 2:
            matrix = data
        else:
            raise ValueError("Input data must be 1-D or 2-D")

        if matrix.shape[0] < 2:
            # PCA needs two samples; a single row is stored exactly as the mean
            return (np.empty((1, 0)), np.empty((0, matrix.shape[1])),
                    matrix[0].astype(float), data.shape)

        n_components = self.n_components
        if isinstance(n_components, (int, np.integer)):
            n_components = min(n_components, *matrix.shape)

        pca = PCA(n_components=n_components)
        scores = pca.fit_transform(matrix)
        return scores, pca.components_, pca.mean_, data.shape

    def decompress(self, compressed_data: tuple) -> np.ndarray:
        if not isinstance(compressed_data, tuple):
            raise TypeError("Input data must be a (scores, components, mean, original_shape) tuple")

        scores, components, mean, original_shape = compressed_data
        reconstructed = scores @ components + mean

        if len(original_shape) == 1:
            # Undo the windowing and drop the padding
            return reconstructed.ravel()[:original_shape[0]]
        return reconstructed

    def __repr__(self):
        return f"PCACompression(n_components={self.n_components}, window_size={self.window_size})"

class TimeSeriesCompressor:
    """Enhanced time series compressor with advanced features."""
    
    def __init__(self, algorithm: Optional[CompressionAlgorithm] = None):
        self.algorithm = algorithm or DifferenceEncoding()
        self.benchmark_results = pd.DataFrame(
            columns=['Algorithm', 'Compression_Ratio', 'MSE', 'Max_Error',
                    'SNR_dB', 'Compression_Time', 'Decompression_Time']
        )

    def set_algorithm(self, algorithm: CompressionAlgorithm) -> None:
        if not isinstance(algorithm, CompressionAlgorithm):
            raise TypeError("Algorithm must be an instance of CompressionAlgorithm")
        self.algorithm = algorithm

    def compress(self, data: np.ndarray) -> Union[np.ndarray, list, bytes, tuple]:
        if len(data) == 0:
            raise ValueError("Input data cannot be empty")
        return self.algorithm.compress(data)

    def decompress(self, compressed_data: Union[np.ndarray, list, bytes, tuple]) -> np.ndarray:
        return self.algorithm.decompress(compressed_data)

    def compress_parallel(self, data: np.ndarray, chunk_size: int = 1000,
                         max_workers: int = 4) -> list:
        """Parallel compression using multiple processes."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        # Chunks of exactly chunk_size points; the last one holds the remainder
        chunks = np.array_split(data, range(chunk_size, len(data), chunk_size))

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            compressed_chunks = list(executor.map(self.compress, chunks))

        return compressed_chunks

    def decompress_parallel(self, compressed_chunks: list,
                          max_workers: int = 4) -> np.ndarray:
        """Parallel decompression using multiple processes."""
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            decompressed_chunks = list(executor.map(self.decompress, compressed_chunks))
            
        return np.concatenate(decompressed_chunks)

    @staticmethod
    def _estimate_size(obj) -> int:
        """Recursively estimate the byte size of a compressed output."""
        if isinstance(obj, np.ndarray):
            return obj.nbytes
        elif isinstance(obj, bytes):
            return len(obj)
        elif isinstance(obj, (list, tuple)):
            return sum(TimeSeriesCompressor._estimate_size(item) for item in obj)
        elif isinstance(obj, np.generic):
            return obj.nbytes
        elif isinstance(obj, (int, float)):
            # Count Python numbers as 8-byte values rather than object overhead
            return 8
        else:
            # Scalar or other primitive
            return sys.getsizeof(obj)

    def benchmark_algorithm(self, data: np.ndarray,
                          algorithm: CompressionAlgorithm) -> dict:
        """Benchmark a compression algorithm's performance."""
        self.set_algorithm(algorithm)

        # Measure compression
        start_time = time.perf_counter()
        compressed = self.compress(data)
        compression_time = time.perf_counter() - start_time

        # Measure decompression
        start_time = time.perf_counter()
        decompressed = self.decompress(compressed)
        decompression_time = time.perf_counter() - start_time

        # Calculate metrics
        compressed_size = self._estimate_size(compressed)

        compression_ratio = data.nbytes / compressed_size
        mse = np.mean((data - decompressed) ** 2)
        max_error = np.max(np.abs(data - decompressed))
        # SNR in dB; guard against zero-signal edge case
        signal_power = np.mean(data ** 2)
        snr_db = 10 * np.log10(signal_power / mse) if mse > 0 else float('inf')

        return {
            'Algorithm': algorithm.__class__.__name__,
            'Compression_Ratio': compression_ratio,
            'MSE': mse,
            'Max_Error': max_error,
            'SNR_dB': snr_db,
            'Compression_Time': compression_time,
            'Decompression_Time': decompression_time
        }

    def benchmark_all(self, data: np.ndarray, 
                     algorithms: List[CompressionAlgorithm]) -> pd.DataFrame:
        """Benchmark multiple compression algorithms."""
        results = []
        for algorithm in algorithms:
            result = self.benchmark_algorithm(data, algorithm)
            results.append(result)
            
        self.benchmark_results = pd.DataFrame(results)
        return self.benchmark_results

    def auto_select_algorithm(self, data: np.ndarray, 
                            algorithms: List[CompressionAlgorithm],
                            priority: str = 'balanced') -> CompressionAlgorithm:
        """
        Automatically select the best algorithm based on benchmarks.
        
        Priority options:
        - 'size': Prioritize compression ratio
        - 'speed': Prioritize processing speed
        - 'accuracy': Prioritize low MSE
        - 'balanced': Consider all factors
        """
        if priority not in ['size', 'speed', 'accuracy', 'balanced']:
            raise ValueError(f"Invalid priority: {priority}")
            
        results = self.benchmark_all(data, algorithms)

        # Normalize metrics (guard against division by zero when all values are equal)
        normalized = results.copy()
        for col in ['Compression_Ratio', 'MSE', 'Compression_Time', 'Decompression_Time']:
            col_range = results[col].max() - results[col].min()
            if col_range == 0:
                normalized[col] = 0.0
            else:
                normalized[col] = (results[col] - results[col].min()) / col_range
        
        # Calculate scores based on priority
        if priority == 'size':
            scores = normalized['Compression_Ratio']
        elif priority == 'speed':
            scores = -(normalized['Compression_Time'] + normalized['Decompression_Time']) / 2
        elif priority == 'accuracy':
            scores = -normalized['MSE']
        else:  # balanced
            scores = (normalized['Compression_Ratio'] 
                     - normalized['MSE'] 
                     - (normalized['Compression_Time'] + normalized['Decompression_Time']) / 4)
        
        best_idx = scores.argmax()
        return algorithms[best_idx]
