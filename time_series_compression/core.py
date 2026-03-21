import numpy as np
import pandas as pd
import time as _time
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Optional, Union, List

from .serialization import serialize_compressed_result, deserialize_compressed_result


@dataclass
class CompressedResult:
    """Container for compressed time series data with all metadata needed for decompression."""
    algorithm: str
    original_shape: tuple
    original_dtype: str
    params: dict
    payload: bytes

    def to_bytes(self) -> bytes:
        """Serialize to a portable binary format."""
        return serialize_compressed_result(self)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'CompressedResult':
        """Deserialize from binary format."""
        return deserialize_compressed_result(data)

    @property
    def compressed_size(self) -> int:
        """Size of the full serialized form in bytes."""
        return len(self.to_bytes())

    @property
    def original_size(self) -> int:
        """Size of the original data in bytes."""
        dtype = np.dtype(self.original_dtype)
        n_elements = 1
        for dim in self.original_shape:
            n_elements *= dim
        return n_elements * dtype.itemsize

    @property
    def compression_ratio(self) -> float:
        return self.original_size / self.compressed_size


class CompressionAlgorithm(ABC):
    """Base class for all compression algorithms."""

    @abstractmethod
    def compress(self, data: np.ndarray) -> CompressedResult:
        pass

    @abstractmethod
    def decompress(self, result: CompressedResult) -> np.ndarray:
        pass

    def _validate_input(self, data: np.ndarray) -> None:
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a numpy array")
        if data.size == 0:
            raise ValueError("Input data cannot be empty")

    def _validate_result(self, result: CompressedResult) -> None:
        if not isinstance(result, CompressedResult):
            raise TypeError("Expected a CompressedResult object")
        if result.algorithm != self.__class__.__name__:
            raise ValueError(
                f"CompressedResult was produced by {result.algorithm}, "
                f"not {self.__class__.__name__}"
            )


class StreamingCompressionAlgorithm(CompressionAlgorithm, ABC):
    """Base class for streaming compression algorithms that process data in chunks."""

    @abstractmethod
    def partial_compress(self, chunk: np.ndarray) -> Optional[bytes]:
        """Process a chunk of data. May return bytes of completed runs."""
        pass

    @abstractmethod
    def finalize_compression(self) -> CompressedResult:
        """Finalize and return the complete CompressedResult."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal streaming state for a new compression session."""
        pass


# Algorithm registry — populated by algorithms.py and streaming.py at import time
_ALGORITHM_REGISTRY: dict = {}


def register_algorithm(cls):
    """Decorator to register an algorithm class by name."""
    _ALGORITHM_REGISTRY[cls.__name__] = cls
    return cls


class TimeSeriesCompressor:
    """Production time series compressor with benchmarking and auto-selection."""

    def __init__(self, algorithm: Optional[CompressionAlgorithm] = None):
        # Deferred import to avoid circular dependency
        from .algorithms import DifferenceEncoding
        self.algorithm = algorithm or DifferenceEncoding()
        self.benchmark_results = pd.DataFrame(
            columns=['Algorithm', 'Compression_Ratio', 'MSE', 'Max_Error',
                     'SNR_dB', 'Compression_Time', 'Decompression_Time',
                     'Compressed_Bytes']
        )

    def set_algorithm(self, algorithm: CompressionAlgorithm) -> None:
        if not isinstance(algorithm, CompressionAlgorithm):
            raise TypeError("Algorithm must be an instance of CompressionAlgorithm")
        self.algorithm = algorithm

    def compress(self, data: np.ndarray) -> CompressedResult:
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a numpy array")
        if data.size == 0:
            raise ValueError("Input data cannot be empty")
        return self.algorithm.compress(data)

    def decompress(self, result: CompressedResult) -> np.ndarray:
        return self.algorithm.decompress(result)

    def compress_to_bytes(self, data: np.ndarray) -> bytes:
        """Compress and serialize to portable binary format."""
        return self.compress(data).to_bytes()

    @staticmethod
    def decompress_from_bytes(data: bytes) -> np.ndarray:
        """Deserialize and decompress. Auto-detects the algorithm used."""
        cr = CompressedResult.from_bytes(data)
        if cr.algorithm not in _ALGORITHM_REGISTRY:
            raise ValueError(f"Unknown algorithm: {cr.algorithm}")
        # Reconstruct algorithm from params
        algo_cls = _ALGORITHM_REGISTRY[cr.algorithm]
        algo = algo_cls(**cr.params.get('_constructor_args', {}))
        return algo.decompress(cr)

    def compress_parallel(self, data: np.ndarray, chunk_size: int = 1000,
                          max_workers: int = 4) -> List[CompressedResult]:
        """Parallel compression by splitting data into chunks."""
        chunks = np.array_split(data, max(1, len(data) // chunk_size))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self.algorithm.compress, chunks))
        return results

    def decompress_parallel(self, results: List[CompressedResult],
                            max_workers: int = 4) -> np.ndarray:
        """Parallel decompression of chunks."""
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            chunks = list(executor.map(self.algorithm.decompress, results))
        return np.concatenate(chunks)

    def benchmark_algorithm(self, data: np.ndarray,
                            algorithm: CompressionAlgorithm) -> dict:
        """Benchmark a compression algorithm's performance."""
        self.set_algorithm(algorithm)

        start = _time.time()
        compressed = self.compress(data)
        compression_time = _time.time() - start

        start = _time.time()
        decompressed = self.decompress(compressed)
        decompression_time = _time.time() - start

        compressed_bytes = len(compressed.to_bytes())
        compression_ratio = data.nbytes / compressed_bytes
        mse = float(np.mean((data - decompressed) ** 2))
        max_error = float(np.max(np.abs(data - decompressed)))
        signal_power = float(np.mean(data ** 2))
        snr_db = 10 * np.log10(signal_power / mse) if mse > 0 else float('inf')

        return {
            'Algorithm': algorithm.__class__.__name__,
            'Compression_Ratio': compression_ratio,
            'MSE': mse,
            'Max_Error': max_error,
            'SNR_dB': snr_db,
            'Compression_Time': compression_time,
            'Decompression_Time': decompression_time,
            'Compressed_Bytes': compressed_bytes,
        }

    def benchmark_all(self, data: np.ndarray,
                      algorithms: List[CompressionAlgorithm]) -> pd.DataFrame:
        """Benchmark multiple compression algorithms."""
        results = [self.benchmark_algorithm(data, algo) for algo in algorithms]
        self.benchmark_results = pd.DataFrame(results)
        return self.benchmark_results

    def auto_select_algorithm(self, data: np.ndarray,
                              algorithms: List[CompressionAlgorithm],
                              priority: str = 'balanced') -> CompressionAlgorithm:
        """Automatically select the best algorithm based on benchmarks.

        Priority options: 'size', 'speed', 'accuracy', 'balanced'
        """
        if priority not in ('size', 'speed', 'accuracy', 'balanced'):
            raise ValueError(f"Invalid priority: {priority}")

        results = self.benchmark_all(data, algorithms)

        normalized = results.copy()
        for col in ['Compression_Ratio', 'MSE', 'Compression_Time', 'Decompression_Time']:
            col_range = results[col].max() - results[col].min()
            normalized[col] = (results[col] - results[col].min()) / col_range if col_range > 0 else 0.0

        if priority == 'size':
            scores = normalized['Compression_Ratio']
        elif priority == 'speed':
            scores = -(normalized['Compression_Time'] + normalized['Decompression_Time']) / 2
        elif priority == 'accuracy':
            scores = -normalized['MSE']
        else:
            scores = (normalized['Compression_Ratio']
                      - normalized['MSE']
                      - (normalized['Compression_Time'] + normalized['Decompression_Time']) / 4)

        return algorithms[scores.argmax()]
