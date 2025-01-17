import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Optional, Union, List, Tuple
from abc import ABC, abstractmethod
from time_series_compression import CompressionAlgorithm, TimeSeriesCompressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import time

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

    def compress(self, data: np.ndarray) -> List[Tuple[float, int, float]]:
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a numpy array")
        
        self.original_shape = data.shape
        compressed = []
        
        # Calculate deltas
        deltas = np.diff(data)
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

    def decompress(self, compressed_data: List[Tuple[float, int, float]]) -> np.ndarray:
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

    def compress(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a numpy array")
            
        self.original_shape = data.shape
        
        # Standardize the data
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        
        # Apply PCA
        compressed_data = self.pca.fit_transform(scaled_data)
        
        return (compressed_data, 
                self.pca.components_, 
                np.array([self.scaler.mean_[0], self.scaler.scale_[0]]))

    def decompress(self, compressed_data: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        data, components, scaler_params = compressed_data
        mean, scale = scaler_params
        
        # Reverse PCA
        reconstructed = np.dot(data, components)
        
        # Reverse standardization
        reconstructed = reconstructed * scale + mean
        
        return reconstructed.reshape(self.original_shape)

class EnhancedTimeSeriesCompressor(TimeSeriesCompressor):
    """
    Enhanced version of TimeSeriesCompressor with additional features:
    - Parallel processing support
    - Automatic algorithm selection
    - Benchmarking capabilities
    - Streaming data support
    """
    
    def __init__(self, algorithm: Optional[CompressionAlgorithm] = None):
        super().__init__(algorithm)
        self.benchmark_results = pd.DataFrame(
            columns=['Algorithm', 'Compression_Ratio', 'MSE', 'Compression_Time', 'Decompression_Time']
        )

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

    def benchmark_algorithm(self, data: np.ndarray, algorithm: CompressionAlgorithm) -> dict:
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

    def benchmark_all(self, data: np.ndarray, algorithms: List[CompressionAlgorithm]) -> pd.DataFrame:
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
        results = self.benchmark_all(data, algorithms)
        
        # Normalize metrics
        normalized = results.copy()
        for col in ['Compression_Ratio', 'MSE', 'Compression_Time', 'Decompression_Time']:
            normalized[col] = (results[col] - results[col].min()) / (results[col].max() - results[col].min())
        
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

# Example usage:
if __name__ == "__main__":
    # Generate sample data
    time = np.arange(0, 10, 0.01)
    data = np.sin(time) + np.random.normal(0, 0.1, time.shape)
    
    # Create compressor instance
    compressor = EnhancedTimeSeriesCompressor()
    
    # Define algorithms to test
    algorithms = [
        DeltaRLE(tolerance=1e-6),
        PCACompression(n_components=0.95)
    ]
    
    # Benchmark algorithms
    results = compressor.benchmark_all(data, algorithms)
    print("\nBenchmark Results:")
    print(results)
    
    # Auto-select best algorithm
    best_algo = compressor.auto_select_algorithm(data, algorithms, priority='balanced')
    print(f"\nBest algorithm selected: {best_algo.__class__.__name__}")
    
    # Test parallel compression
    compressed_chunks = compressor.compress_parallel(data, chunk_size=1000)
    decompressed_data = compressor.decompress_parallel(compressed_chunks)
    
    print(f"\nParallel processing MSE: {np.mean((data - decompressed_data) ** 2)}")
