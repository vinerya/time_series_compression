import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import norm
from scipy.fft import dct, idct
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import zlib
import pywt
import time

class CompressionAlgorithm(ABC):
    """Base class for all compression algorithms."""
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

class PAA(CompressionAlgorithm):
    def __init__(self, segments):
        self.segments = segments

    def compress(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a numpy array")
        
        self.original_length = len(data)
        segment_len = len(data) // self.segments
        compressed = np.mean(data[:len(data) - len(data) % segment_len].reshape(-1, segment_len), axis=1)
        return compressed

    def decompress(self, compressed_data):
        if not isinstance(compressed_data, np.ndarray):
            raise TypeError("Input data must be a numpy array")
        
        return np.repeat(compressed_data, self.original_length // self.segments)

class SAX(CompressionAlgorithm):
    def __init__(self, segments, alphabet_size):
        self.segments = segments
        self.alphabet_size = alphabet_size
        self.breakpoints = norm.ppf(np.linspace(0, 1, alphabet_size + 1)[1:-1])

    def compress(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a numpy array")
        
        self.original_length = len(data)
        self.original_mean = np.mean(data)
        self.original_std = np.std(data)
        
        # Normalize the data
        normalized_data = (data - self.original_mean) / self.original_std
        
        # PAA compression
        paa = PAA(self.segments)
        paa_data = paa.compress(normalized_data)
        
        # Discretize to symbols
        symbolic_data = np.digitize(paa_data, self.breakpoints)
        return symbolic_data

    def decompress(self, compressed_data):
        if not isinstance(compressed_data, np.ndarray):
            raise TypeError("Input data must be a numpy array")
        
        # Convert symbols back to PAA
        paa_data = self.breakpoints[compressed_data - 1]
        
        # PAA decompression
        paa = PAA(self.segments)
        paa.original_length = self.original_length
        decompressed = paa.decompress(paa_data)
        
        # Denormalize
        return decompressed * self.original_std + self.original_mean

class DCT(CompressionAlgorithm):
    def __init__(self, keep_coeffs):
        self.keep_coeffs = keep_coeffs

    def compress(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a numpy array")
        
        self.original_shape = data.shape
        dct_coeffs = dct(data)
        compressed = dct_coeffs[:self.keep_coeffs]
        return compressed

    def decompress(self, compressed_data):
        if not isinstance(compressed_data, np.ndarray):
            raise TypeError("Input data must be a numpy array")
        
        full_coeffs = np.zeros(self.original_shape)
        full_coeffs[:len(compressed_data)] = compressed_data
        return idct(full_coeffs, n=self.original_shape[0])

class RunLengthEncoding(CompressionAlgorithm):
    def compress(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a numpy array")
        
        self.original_shape = data.shape
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
        return np.array(decompressed).reshape(self.original_shape)

class ZlibCompression(CompressionAlgorithm):
    def compress(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a numpy array")
        
        self.original_shape = data.shape
        return zlib.compress(data.tobytes())

    def decompress(self, compressed_data):
        decompressed = zlib.decompress(compressed_data)
        return np.frombuffer(decompressed, dtype=np.float64).reshape(self.original_shape)

class DiscreteWaveletTransform(CompressionAlgorithm):
    def __init__(self, wavelet='db4', level=None, threshold=0.1):
        self.wavelet = wavelet
        self.level = level
        self.threshold = threshold

    def compress(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a numpy array")
        
        self.original_shape = data.shape
        coeffs = pywt.wavedec(data, self.wavelet, level=self.level)
        
        # Threshold the coefficients
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], self.threshold * np.max(np.abs(coeffs[i])))
        
        return coeffs

    def decompress(self, compressed_data):
        if not isinstance(compressed_data, list):
            raise TypeError("Input data must be a list of wavelet coefficients")
        
        return pywt.waverec(compressed_data, self.wavelet)[:self.original_shape[0]]

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
    """
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance

    def compress(self, data: np.ndarray) -> list:
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a numpy array")
        if len(data) == 0:
            raise ValueError("Input data cannot be empty")
        
        self.original_shape = data.shape
        compressed = []
        
        # Calculate deltas
        deltas = np.diff(data)
        if len(deltas) == 0:
            return [(data[0], 1, 0.0)]
            
        current_delta = deltas[0]
        count = 1
        start_val = data[0]
        
        # Compress runs of similar deltas
        for i in range(1, len(deltas)):
            if abs(deltas[i] - current_delta) < self.tolerance:
                count += 1
            else:
                compressed.append((start_val, count, current_delta))
                start_val = data[i]
                current_delta = deltas[i]
                count = 1
                
        compressed.append((start_val, count, current_delta))
        return compressed

    def decompress(self, compressed_data: list) -> np.ndarray:
        decompressed = []
        
        for start_val, count, delta in compressed_data:
            values = [start_val + i * delta for i in range(count + 1)]
            decompressed.extend(values)
            
        return np.array(decompressed[:-len(compressed_data)+1])

class PCACompression(CompressionAlgorithm):
    """
    PCA-based compression algorithm.
    Effective for high-dimensional time series with correlated components.
    """
    
    def __init__(self, n_components: Optional[Union[int, float]] = 0.95):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()

    def compress(self, data: np.ndarray) -> tuple:
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a numpy array")
        if len(data) == 0:
            raise ValueError("Input data cannot be empty")
            
        self.original_shape = data.shape
        
        # Standardize the data
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        
        # Apply PCA
        compressed_data = self.pca.fit_transform(scaled_data)
        
        return (compressed_data, 
                self.pca.components_, 
                np.array([self.scaler.mean_[0], self.scaler.scale_[0]]))

    def decompress(self, compressed_data: tuple) -> np.ndarray:
        data, components, scaler_params = compressed_data
        mean, scale = scaler_params
        
        # Reverse PCA
        reconstructed = np.dot(data, components)
        
        # Reverse standardization
        reconstructed = reconstructed * scale + mean
        
        return reconstructed.reshape(self.original_shape)

class TimeSeriesCompressor:
    """Enhanced time series compressor with advanced features."""
    
    def __init__(self, algorithm: Optional[CompressionAlgorithm] = None):
        self.algorithm = algorithm or DifferenceEncoding()
        self.benchmark_results = pd.DataFrame(
            columns=['Algorithm', 'Compression_Ratio', 'MSE', 
                    'Compression_Time', 'Decompression_Time']
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
                         max_workers: int = 4) -> List[np.ndarray]:
        """Parallel compression using multiple processes."""
        chunks = np.array_split(data, len(data) // chunk_size + 1)
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            compressed_chunks = list(executor.map(self.compress, chunks))
            
        return compressed_chunks

    def decompress_parallel(self, compressed_chunks: List[np.ndarray], 
                          max_workers: int = 4) -> np.ndarray:
        """Parallel decompression using multiple processes."""
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            decompressed_chunks = list(executor.map(self.decompress, compressed_chunks))
            
        return np.concatenate(decompressed_chunks)

    def benchmark_algorithm(self, data: np.ndarray, 
                          algorithm: CompressionAlgorithm) -> dict:
        """Benchmark a compression algorithm's performance."""
        self.set_algorithm(algorithm)
        
        # Measure compression
        start_time = time.time()
        compressed = self.compress(data)
        compression_time = time.time() - start_time
        
        # Measure decompression
        start_time = time.time()
        decompressed = self.decompress(compressed)
        decompression_time = time.time() - start_time
        
        # Calculate metrics
        if isinstance(compressed, list):
            compressed_size = sum(arr.nbytes if isinstance(arr, np.ndarray) 
                                else len(str(arr)) for arr in compressed)
        elif isinstance(compressed, bytes):
            compressed_size = len(compressed)
        else:
            compressed_size = compressed.nbytes
            
        compression_ratio = data.nbytes / compressed_size
        mse = np.mean((data - decompressed) ** 2)
        
        return {
            'Algorithm': algorithm.__class__.__name__,
            'Compression_Ratio': compression_ratio,
            'MSE': mse,
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
        
        # Normalize metrics
        normalized = results.copy()
        for col in ['Compression_Ratio', 'MSE', 'Compression_Time', 'Decompression_Time']:
            normalized[col] = ((results[col] - results[col].min()) / 
                             (results[col].max() - results[col].min()))
        
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
