import numpy as np
import matplotlib.pyplot as plt
from time_series_compression import (
    TimeSeriesCompressor, DifferenceEncoding, PAA, SAX, DCT,
    RunLengthEncoding, ZlibCompression, DiscreteWaveletTransform,
    DeltaRLE, PCACompression
)

def generate_sample_data(size=1000):
    """Generate sample time series data."""
    time = np.arange(0, 10, 10/size)
    return time, np.sin(time) + np.random.normal(0, 0.1, time.shape)

def basic_compression_example():
    """Demonstrate basic compression functionality."""
    print("\n1. Basic Compression Example")
    print("----------------------------")
    
    time, data = generate_sample_data()
    compressor = TimeSeriesCompressor()
    
    algorithms = [
        ("DifferenceEncoding (Lossless)", DifferenceEncoding()),
        ("PAA (Lossy)", PAA(segments=50)),
        ("SAX (Lossy)", SAX(segments=50, alphabet_size=10)),
        ("DCT (Lossy)", DCT(keep_coeffs=50)),
        ("RunLengthEncoding (Lossless)", RunLengthEncoding()),
        ("ZlibCompression (Lossless)", ZlibCompression()),
        ("DiscreteWaveletTransform (Lossy)", DiscreteWaveletTransform(wavelet='db4', level=5, threshold=0.1))
    ]
    
    for name, algo in algorithms:
        compressor.set_algorithm(algo)
        compressed = compressor.compress(data)
        decompressed = compressor.decompress(compressed)
        mse = np.mean((data - decompressed) ** 2)
        print(f"\n{name}:")
        print(f"MSE: {mse:.6f}")

def enhanced_features_example():
    """Demonstrate new enhanced features."""
    print("\n2. Enhanced Features Example")
    print("---------------------------")
    
    # Generate larger dataset for parallel processing demo
    time, data = generate_sample_data(size=100000)
    
    # Initialize enhanced compressor
    enhanced_compressor = TimeSeriesCompressor()
    
    # Define algorithms including new ones
    algorithms = [
        DeltaRLE(tolerance=1e-6),
        PCACompression(n_components=0.95),
        DifferenceEncoding(),
        PAA(segments=100)
    ]
    
    print("\nA. Automatic Algorithm Selection")
    print("--------------------------------")
    # Try different priorities
    for priority in ['size', 'speed', 'accuracy', 'balanced']:
        best_algo = enhanced_compressor.auto_select_algorithm(data, algorithms, priority=priority)
        print(f"Best algorithm for {priority} priority: {best_algo.__class__.__name__}")
    
    print("\nB. Parallel Processing")
    print("---------------------")
    # Compare parallel vs sequential processing
    import time
    
    # Sequential processing
    start_time = time.time()
    enhanced_compressor.set_algorithm(DeltaRLE())
    compressed = enhanced_compressor.compress(data)
    seq_time = time.time() - start_time
    
    # Parallel processing
    start_time = time.time()
    compressed_chunks = enhanced_compressor.compress_parallel(data, chunk_size=1000)
    par_time = time.time() - start_time
    
    print(f"Sequential processing time: {seq_time:.3f}s")
    print(f"Parallel processing time: {par_time:.3f}s")
    print(f"Speedup: {seq_time/par_time:.2f}x")
    
    print("\nC. Benchmarking")
    print("--------------")
    results = enhanced_compressor.benchmark_all(data[:10000], algorithms)
    print("\nBenchmark Results:")
    print(results)

def plot_comparison():
    """Create comparison plots of different algorithms."""
    print("\n3. Plotting Comparison")
    print("---------------------")
    
    time, data = generate_sample_data()
    enhanced_compressor = TimeSeriesCompressor()
    
    plt.figure(figsize=(15, 10))
    plt.plot(time, data, label='Original Data', alpha=0.5)
    
    algorithms = [
        DeltaRLE(tolerance=1e-6),
        PCACompression(n_components=0.95),
        PAA(segments=50),
        DCT(keep_coeffs=50)
    ]
    
    for algo in algorithms:
        enhanced_compressor.set_algorithm(algo)
        decompressed = enhanced_compressor.decompress(enhanced_compressor.compress(data))
        plt.plot(time[:len(decompressed)], decompressed, 
                label=f'Decompressed ({algo.__class__.__name__})',
                alpha=0.7)
    
    plt.legend()
    plt.title('Comparison of Compression Algorithms')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.savefig('enhanced_compression_comparison.png')
    plt.close()
    
    print("Plot saved as 'enhanced_compression_comparison.png'")

if __name__ == "__main__":
    basic_compression_example()
    enhanced_features_example()
    plot_comparison()
