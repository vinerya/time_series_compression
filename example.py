"""Example usage of the time_series_compression library."""

import numpy as np
from time_series_compression import (
    TimeSeriesCompressor, CompressedResult,
    DifferenceEncoding, PAA, SAX, DCT,
    RunLengthEncoding, ZlibCompression, DiscreteWaveletTransform,
    DeltaRLE, StreamingDeltaRLE,
)


def generate_sample_data(size=1000):
    rng = np.random.default_rng(42)
    t = np.linspace(0, 10, size)
    return t, np.sin(t) + rng.normal(0, 0.1, size)


def basic_compression():
    """Compress → serialize → deserialize → decompress."""
    print("\n1. Basic Compression & Byte Serialization")
    print("-" * 45)

    _, data = generate_sample_data()
    compressor = TimeSeriesCompressor()

    algorithms = [
        ("DifferenceEncoding (lossless)", DifferenceEncoding()),
        ("PAA (lossy)", PAA(segments=50)),
        ("SAX (lossy)", SAX(segments=50, alphabet_size=10)),
        ("DCT (lossy)", DCT(keep_coeffs=50)),
        ("RLE (lossless)", RunLengthEncoding()),
        ("Zlib (lossless)", ZlibCompression()),
        ("DWT (lossy)", DiscreteWaveletTransform(wavelet='db4', level=5, threshold=0.1)),
        ("DeltaRLE (lossless)", DeltaRLE()),
    ]

    for name, algo in algorithms:
        compressor.set_algorithm(algo)
        result = compressor.compress(data)

        # Serialize to bytes and back
        raw_bytes = result.to_bytes()
        restored = CompressedResult.from_bytes(raw_bytes)
        decompressed = compressor.decompress(restored)

        mse = np.mean((data - decompressed) ** 2)
        ratio = data.nbytes / len(raw_bytes)
        print(f"\n  {name}:")
        print(f"    Serialized size: {len(raw_bytes):,} bytes "
              f"(original: {data.nbytes:,} bytes)")
        print(f"    Compression ratio: {ratio:.2f}x")
        print(f"    MSE: {mse:.8f}")


def auto_decompress_demo():
    """Compress to bytes and auto-decompress without knowing the algorithm."""
    print("\n\n2. Auto-Detect Decompression from Bytes")
    print("-" * 45)

    _, data = generate_sample_data()
    compressor = TimeSeriesCompressor(ZlibCompression())

    raw_bytes = compressor.compress_to_bytes(data)
    print(f"  Compressed {data.nbytes:,} bytes → {len(raw_bytes):,} bytes")

    # Decompress without specifying algorithm — it's encoded in the bytes
    restored = TimeSeriesCompressor.decompress_from_bytes(raw_bytes)
    print(f"  Decompressed back to {restored.nbytes:,} bytes")
    print(f"  Perfect reconstruction: {np.array_equal(data, restored)}")


def streaming_demo():
    """Process data in chunks using streaming compression."""
    print("\n\n3. Streaming Compression")
    print("-" * 45)

    _, data = generate_sample_data(size=10000)
    chunk_size = 1000

    algo = StreamingDeltaRLE(tolerance=1e-6)
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        partial = algo.partial_compress(chunk)
        if partial:
            print(f"  Chunk {i // chunk_size + 1}: flushed {len(partial)} bytes")

    result = algo.finalize_compression()
    print(f"\n  Total compressed: {len(result.to_bytes()):,} bytes "
          f"(original: {data.nbytes:,} bytes)")

    decompressed = algo.decompress(result)
    mse = np.mean((data - decompressed) ** 2)
    print(f"  MSE: {mse:.10f}")


def benchmarking_demo():
    """Benchmark and auto-select the best algorithm."""
    print("\n\n4. Benchmarking & Auto-Selection")
    print("-" * 45)

    _, data = generate_sample_data(size=10000)
    compressor = TimeSeriesCompressor()

    algorithms = [
        DeltaRLE(),
        DifferenceEncoding(),
        PAA(segments=100),
        DCT(keep_coeffs=100),
        ZlibCompression(),
    ]

    results = compressor.benchmark_all(data, algorithms)
    print("\n  Benchmark Results:")
    print(results.to_string(index=False))

    for priority in ['size', 'speed', 'accuracy', 'balanced']:
        best = compressor.auto_select_algorithm(data, algorithms, priority=priority)
        print(f"\n  Best for {priority}: {best!r}")


def plot_comparison():
    """Visual comparison (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n\n5. Skipping plot (install matplotlib: pip install matplotlib)")
        return

    print("\n\n5. Plotting Comparison")
    print("-" * 45)

    t, data = generate_sample_data()
    compressor = TimeSeriesCompressor()

    plt.figure(figsize=(15, 10))
    plt.plot(t, data, label='Original', alpha=0.5)

    for algo in [DeltaRLE(), PAA(segments=50), DCT(keep_coeffs=50)]:
        compressor.set_algorithm(algo)
        result = compressor.compress(data)
        decompressed = compressor.decompress(result)
        plt.plot(t[:len(decompressed)], decompressed,
                 label=repr(algo), alpha=0.7)

    plt.legend()
    plt.title('Compression Algorithm Comparison')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.savefig('compression_comparison.png')
    plt.close()
    print("  Saved: compression_comparison.png")


if __name__ == "__main__":
    basic_compression()
    auto_decompress_demo()
    streaming_demo()
    benchmarking_demo()
    plot_comparison()
