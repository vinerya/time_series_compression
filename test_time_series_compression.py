import pickle
import unittest
import warnings
import numpy as np
import pandas as pd
from time_series_compression import (
    TimeSeriesCompressor, DifferenceEncoding, PAA, SAX, DCT,
    RunLengthEncoding, ZlibCompression, DiscreteWaveletTransform,
    DeltaRLE, PCACompression, StreamingCompressionAlgorithm
)

class TestBaseCompressor(unittest.TestCase):
    def setUp(self):
        self.compressor = TimeSeriesCompressor()
        self.time = np.arange(0, 10, 0.1)
        rng = np.random.default_rng(42)
        self.data = np.sin(self.time) + rng.normal(0, 0.1, self.time.shape)

    def test_default_algorithm(self):
        self.assertIsInstance(self.compressor.algorithm, DifferenceEncoding)

    def test_set_algorithm(self):
        new_algorithm = PAA(segments=10)
        self.compressor.set_algorithm(new_algorithm)
        self.assertIs(self.compressor.algorithm, new_algorithm)

    def test_compress_decompress_difference_encoding(self):
        self.compressor.set_algorithm(DifferenceEncoding())
        compressed_data = self.compressor.compress(self.data)
        decompressed_data = self.compressor.decompress(compressed_data)

        self.assertEqual(self.data.shape, decompressed_data.shape)
        self.assertFalse(np.array_equal(self.data, compressed_data))
        np.testing.assert_allclose(self.data, decompressed_data, rtol=1e-10, atol=1e-10)

    def test_compress_decompress_paa(self):
        self.compressor.set_algorithm(PAA(segments=10))
        compressed_data = self.compressor.compress(self.data)
        decompressed_data = self.compressor.decompress(compressed_data)

        means, original_length = compressed_data
        self.assertEqual(means.shape[0], 10)
        self.assertEqual(original_length, len(self.data))
        self.assertEqual(self.data.shape, decompressed_data.shape)
        # PAA with 10 segments on 100 points is very lossy; verify MSE is reasonable
        mse = np.mean((self.data - decompressed_data) ** 2)
        self.assertLess(mse, 0.5)

    def test_compress_decompress_sax(self):
        self.compressor.set_algorithm(SAX(segments=10, alphabet_size=5))
        compressed_data = self.compressor.compress(self.data)
        decompressed_data = self.compressor.decompress(compressed_data)

        symbols = compressed_data[0]
        self.assertEqual(symbols.shape[0], 10)
        self.assertTrue(np.all((symbols >= 0) & (symbols < 5)))
        self.assertEqual(self.data.shape, decompressed_data.shape)
        mse = np.mean((self.data - decompressed_data) ** 2)
        self.assertLess(mse, 1.0)

    def test_compress_decompress_dct(self):
        self.compressor.set_algorithm(DCT(keep_coeffs=10))
        compressed_data = self.compressor.compress(self.data)
        decompressed_data = self.compressor.decompress(compressed_data)

        self.assertEqual(compressed_data[0].shape[0], 10)
        self.assertEqual(self.data.shape, decompressed_data.shape)
        mse = np.mean((self.data - decompressed_data) ** 2)
        self.assertLess(mse, 0.5)

    def test_compress_decompress_run_length_encoding(self):
        self.compressor.set_algorithm(RunLengthEncoding())
        compressed_data = self.compressor.compress(self.data)
        decompressed_data = self.compressor.decompress(compressed_data)

        self.assertIsInstance(compressed_data, list)
        self.assertEqual(self.data.shape, decompressed_data.shape)
        np.testing.assert_allclose(self.data, decompressed_data, rtol=1e-10, atol=1e-10)

    def test_compress_decompress_zlib(self):
        self.compressor.set_algorithm(ZlibCompression())
        compressed_data = self.compressor.compress(self.data)
        decompressed_data = self.compressor.decompress(compressed_data)

        self.assertIsInstance(compressed_data, bytes)
        self.assertEqual(self.data.shape, decompressed_data.shape)
        np.testing.assert_allclose(self.data, decompressed_data, rtol=1e-10, atol=1e-10)

    def test_compress_decompress_dwt(self):
        self.compressor.set_algorithm(DiscreteWaveletTransform(wavelet='db4', level=3, threshold=0.1))
        compressed_data = self.compressor.compress(self.data)
        decompressed_data = self.compressor.decompress(compressed_data)

        self.assertIsInstance(compressed_data, tuple)
        self.assertEqual(self.data.shape, decompressed_data.shape)
        np.testing.assert_allclose(self.data, decompressed_data, rtol=1e-1, atol=1e-1)

    def test_compression_ratio(self):
        original_size = self.data.nbytes

        for algorithm in [DifferenceEncoding(), PAA(segments=10), SAX(segments=10, alphabet_size=5), 
                          DCT(keep_coeffs=10), RunLengthEncoding(), ZlibCompression(),
                          DiscreteWaveletTransform(wavelet='db4', level=3, threshold=0.1)]:
            self.compressor.set_algorithm(algorithm)
            compressed_data = self.compressor.compress(self.data)
            compressed_size = TimeSeriesCompressor._estimate_size(compressed_data)

            ratio = original_size / compressed_size
            # RLE expands random float data (no repeated values), so use a lower threshold
            min_ratio = 0.1 if isinstance(algorithm, RunLengthEncoding) else 0.5
            self.assertGreater(ratio, min_ratio, f"{algorithm.__class__.__name__} compression ratio is too low")

    def test_input_validation(self):
        with self.assertRaises(TypeError):
            self.compressor.compress([1, 2, 3, 4, 5])

        with self.assertRaises(TypeError):
            self.compressor.decompress([1, 2, 3, 4, 5])

    def test_invalid_algorithm(self):
        with self.assertRaises(TypeError):
            self.compressor.set_algorithm("not an algorithm")

class TestAdvancedFeatures(unittest.TestCase):
    def setUp(self):
        self.compressor = TimeSeriesCompressor()
        self.time = np.arange(0, 10, 0.1)
        rng = np.random.default_rng(42)
        self.data = np.sin(self.time) + rng.normal(0, 0.1, self.time.shape)

    def test_delta_rle_compression(self):
        self.compressor.set_algorithm(DeltaRLE(tolerance=1e-6))
        compressed_data = self.compressor.compress(self.data)
        decompressed_data = self.compressor.decompress(compressed_data)

        self.assertIsInstance(compressed_data, list)
        self.assertEqual(self.data.shape, decompressed_data.shape)
        np.testing.assert_allclose(self.data, decompressed_data, rtol=1e-6, atol=1e-6)

    def test_pca_compression(self):
        self.compressor.set_algorithm(PCACompression(n_components=0.95))
        compressed_data = self.compressor.compress(self.data)
        decompressed_data = self.compressor.decompress(compressed_data)

        self.assertIsInstance(compressed_data, tuple)
        self.assertEqual(len(compressed_data), 4)  # scores, components, mean, original_shape
        self.assertEqual(self.data.shape, decompressed_data.shape)
        # Lossy: the discarded components are mostly the added noise (std 0.1)
        mse = np.mean((self.data - decompressed_data) ** 2)
        self.assertLess(mse, 0.01)

    def test_parallel_processing(self):
        self.compressor.set_algorithm(DeltaRLE())
        
        # Test parallel compression
        compressed_chunks = self.compressor.compress_parallel(self.data, chunk_size=10)
        self.assertIsInstance(compressed_chunks, list)
        
        # Test parallel decompression
        decompressed_data = self.compressor.decompress_parallel(compressed_chunks)
        self.assertEqual(self.data.shape, decompressed_data.shape)
        np.testing.assert_allclose(self.data, decompressed_data, rtol=1e-6, atol=1e-6)

    def test_auto_algorithm_selection(self):
        algorithms = [
            DeltaRLE(),
            PCACompression(),
            DifferenceEncoding(),
            PAA(segments=10)
        ]
        
        # Test different priorities
        priorities = ['size', 'speed', 'accuracy', 'balanced']
        for priority in priorities:
            best_algo = self.compressor.auto_select_algorithm(
                self.data, algorithms, priority=priority
            )
            self.assertIn(best_algo.__class__, [algo.__class__ for algo in algorithms])

    def test_benchmarking(self):
        algorithms = [DeltaRLE(), PCACompression()]
        results = self.compressor.benchmark_all(self.data, algorithms)

        self.assertIsInstance(results, pd.DataFrame)
        self.assertEqual(len(results), len(algorithms))
        expected_columns = [
            'Algorithm', 'Compression_Ratio', 'MSE', 'Max_Error',
            'SNR_dB', 'Compression_Time', 'Decompression_Time'
        ]
        self.assertTrue(all(col in results.columns for col in expected_columns))

    def test_streaming_base_class(self):
        # Test that StreamingCompressionAlgorithm properly enforces interface
        with self.assertRaises(TypeError):
            # Should raise because abstract methods aren't implemented
            class IncompleteStreamingAlgo(StreamingCompressionAlgorithm):
                pass
            IncompleteStreamingAlgo()

class TestEdgeCases(unittest.TestCase):
    def setUp(self):
        self.compressor = TimeSeriesCompressor()
        
    def test_empty_data(self):
        data = np.array([])
        with self.assertRaises(ValueError):
            self.compressor.set_algorithm(DeltaRLE())
            self.compressor.compress(data)

    def test_single_value(self):
        data = np.array([1.0])
        self.compressor.set_algorithm(DeltaRLE())
        compressed = self.compressor.compress(data)
        decompressed = self.compressor.decompress(compressed)
        np.testing.assert_array_equal(data, decompressed)

    def test_constant_data(self):
        data = np.ones(100)
        self.compressor.set_algorithm(DeltaRLE())
        compressed = self.compressor.compress(data)
        decompressed = self.compressor.decompress(compressed)
        np.testing.assert_array_equal(data, decompressed)
        self.assertLess(len(compressed), len(data))

    def test_invalid_priority(self):
        with self.assertRaises(ValueError):
            self.compressor.auto_select_algorithm(
                np.array([1, 2, 3]),
                [DeltaRLE()],
                priority='invalid_priority'
            )

    def test_parallel_processing_single_chunk(self):
        data = np.array([1.0, 2.0, 3.0])
        self.compressor.set_algorithm(DeltaRLE())
        compressed = self.compressor.compress_parallel(data, chunk_size=5)  # chunk_size > data length
        decompressed = self.compressor.decompress_parallel(compressed)
        np.testing.assert_array_equal(data, decompressed)

class TestImprovements(unittest.TestCase):
    """Tests for bug fixes and new improvements."""

    def setUp(self):
        self.compressor = TimeSeriesCompressor()
        self.time = np.arange(0, 10, 0.1)
        rng = np.random.default_rng(42)
        self.data = np.sin(self.time) + rng.normal(0, 0.1, self.time.shape)

    def test_paa_non_divisible_length(self):
        """PAA should handle data lengths not evenly divisible by segments."""
        data = np.arange(17, dtype=float)  # 17 is not divisible by 5
        paa = PAA(segments=5)
        compressed = paa.compress(data)
        decompressed = paa.decompress(compressed)
        self.assertEqual(len(decompressed), len(data))
        self.assertEqual(compressed[0].shape[0], 5)

    def test_paa_invalid_segments(self):
        """PAA should reject non-positive segments."""
        with self.assertRaises(ValueError):
            PAA(segments=0)
        with self.assertRaises(ValueError):
            PAA(segments=-1)

    def test_sax_invalid_params(self):
        """SAX should reject invalid parameters."""
        with self.assertRaises(ValueError):
            SAX(segments=0, alphabet_size=5)
        with self.assertRaises(ValueError):
            SAX(segments=10, alphabet_size=1)

    def test_dct_invalid_params(self):
        """DCT should reject non-positive keep_coeffs."""
        with self.assertRaises(ValueError):
            DCT(keep_coeffs=0)

    def test_zlib_preserves_dtype(self):
        """ZlibCompression should preserve the original array dtype."""
        for dtype in [np.float32, np.float64, np.int32]:
            data = np.array([1, 2, 3, 4, 5], dtype=dtype)
            algo = ZlibCompression()
            compressed = algo.compress(data)
            decompressed = algo.decompress(compressed)
            self.assertEqual(decompressed.dtype, dtype)
            np.testing.assert_array_equal(data, decompressed)

    def test_benchmark_new_metrics(self):
        """Benchmark should include Max_Error and SNR_dB metrics."""
        result = self.compressor.benchmark_algorithm(self.data, DifferenceEncoding())
        self.assertIn('Max_Error', result)
        self.assertIn('SNR_dB', result)
        self.assertGreaterEqual(result['Max_Error'], 0)

    def test_benchmark_handles_tuple_output(self):
        """Benchmark should work with PCACompression (returns tuples)."""
        result = self.compressor.benchmark_algorithm(self.data, PCACompression())
        self.assertIn('Compression_Ratio', result)
        self.assertGreater(result['Compression_Ratio'], 0)

    def test_auto_select_single_algorithm(self):
        """auto_select should handle single-algorithm list (division by zero guard)."""
        best = self.compressor.auto_select_algorithm(
            self.data, [DifferenceEncoding()], priority='balanced'
        )
        self.assertIsInstance(best, DifferenceEncoding)

    def test_repr_methods(self):
        """All algorithms should have meaningful __repr__."""
        algos = [
            DifferenceEncoding(),
            PAA(segments=10),
            SAX(segments=10, alphabet_size=5),
            DCT(keep_coeffs=10),
            RunLengthEncoding(),
            ZlibCompression(),
            DiscreteWaveletTransform(wavelet='db4', level=3, threshold=0.1),
            DeltaRLE(tolerance=1e-6),
            PCACompression(n_components=0.95),
        ]
        for algo in algos:
            r = repr(algo)
            self.assertIn(algo.__class__.__name__, r)

    def test_estimate_size_various_types(self):
        """_estimate_size should handle ndarray, bytes, list, tuple, and scalars."""
        self.assertEqual(
            TimeSeriesCompressor._estimate_size(np.zeros(10, dtype=np.float64)),
            80
        )
        self.assertEqual(
            TimeSeriesCompressor._estimate_size(b'hello'),
            5
        )
        # Nested structures (tuple of arrays)
        size = TimeSeriesCompressor._estimate_size((np.zeros(5), np.zeros(3)))
        self.assertEqual(size, 5 * 8 + 3 * 8)

    def test_estimate_size_scalars(self):
        """Scalars count as their value size, not Python object overhead."""
        self.assertEqual(TimeSeriesCompressor._estimate_size(np.float64(1.0)), 8)
        self.assertEqual(TimeSeriesCompressor._estimate_size(np.int32(1)), 4)
        self.assertEqual(TimeSeriesCompressor._estimate_size(100), 8)
        self.assertEqual(TimeSeriesCompressor._estimate_size(1.5), 8)

class TestSelfContainedOutput(unittest.TestCase):
    """Compressed output must carry everything needed to decompress it."""

    CONFIGS = [
        (DifferenceEncoding, {}),
        (PAA, dict(segments=10)),
        (SAX, dict(segments=10, alphabet_size=5)),
        (DCT, dict(keep_coeffs=10)),
        (RunLengthEncoding, {}),
        (ZlibCompression, {}),
        (DiscreteWaveletTransform, dict(wavelet='db4', level=3, threshold=0.1)),
        (DeltaRLE, {}),
        (PCACompression, {}),
    ]

    def setUp(self):
        time = np.arange(0, 10, 0.1)
        rng = np.random.default_rng(42)
        self.data = np.sin(time) + rng.normal(0, 0.1, time.shape)

    def test_decompress_with_fresh_instance(self):
        for cls, kwargs in self.CONFIGS:
            with self.subTest(algorithm=cls.__name__):
                algo = cls(**kwargs)
                compressed = algo.compress(self.data)
                expected = algo.decompress(compressed)
                # Round-trip through pickle, as storage or another process would
                restored = pickle.loads(pickle.dumps(compressed))
                np.testing.assert_array_equal(cls(**kwargs).decompress(restored), expected)

    def test_instance_reuse_keeps_earlier_output_valid(self):
        for cls, kwargs in self.CONFIGS:
            with self.subTest(algorithm=cls.__name__):
                algo = cls(**kwargs)
                compressed = algo.compress(self.data)
                expected = algo.decompress(compressed)
                algo.compress(np.arange(37, dtype=float))
                np.testing.assert_array_equal(algo.decompress(compressed), expected)

    def test_parallel_all_algorithms(self):
        compressor = TimeSeriesCompressor()
        for cls, kwargs in self.CONFIGS:
            with self.subTest(algorithm=cls.__name__):
                compressor.set_algorithm(cls(**kwargs))
                chunks = compressor.compress_parallel(self.data, chunk_size=30)
                decompressed = compressor.decompress_parallel(chunks)
                self.assertEqual(decompressed.shape, self.data.shape)
                self.assertLess(np.mean((self.data - decompressed) ** 2), 0.5)

    def test_parallel_chunk_size_respected(self):
        compressor = TimeSeriesCompressor(DifferenceEncoding())
        chunks = compressor.compress_parallel(self.data, chunk_size=10)
        self.assertEqual([len(c) for c in chunks], [10] * 10)
        chunks = compressor.compress_parallel(self.data, chunk_size=30)
        self.assertEqual([len(c) for c in chunks], [30, 30, 30, 10])
        with self.assertRaises(ValueError):
            compressor.compress_parallel(self.data, chunk_size=0)

class TestAlgorithmFixes(unittest.TestCase):
    """Regression tests for SAX decoding, DeltaRLE error bounds and PCA."""

    def test_sax_symbols_decode_in_order(self):
        """Lower symbols must decode to lower values."""
        data = np.repeat([-3.0, 0.0, 1.0, 3.0], 4)
        sax = SAX(segments=4, alphabet_size=4)
        symbols = sax.compress(data)[0]
        np.testing.assert_array_equal(symbols, [0, 1, 2, 3])
        segment_values = sax.decompress(sax.compress(data))[::4]
        self.assertTrue(np.all(np.diff(segment_values) > 0))

    def test_sax_centroids(self):
        """Symbols decode to the mean of the standard normal within their bin."""
        np.testing.assert_allclose(SAX(segments=1, alphabet_size=2).centroids,
                                   [-np.sqrt(2 / np.pi), np.sqrt(2 / np.pi)])
        centroids = SAX(segments=1, alphabet_size=5).centroids
        self.assertAlmostEqual(centroids.sum(), 0.0)
        self.assertTrue(np.all(np.diff(centroids) > 0))

    def test_sax_constant_data(self):
        """Constant data must not divide by zero."""
        data = np.full(8, 3.0)
        sax = SAX(segments=4, alphabet_size=4)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            decompressed = sax.decompress(sax.compress(data))
        np.testing.assert_array_equal(decompressed, data)

    def test_delta_rle_error_bounded_by_tolerance(self):
        """Small per-step jitter must not accumulate beyond the tolerance."""
        rng = np.random.default_rng(0)
        data = np.cumsum(1.0 + rng.uniform(-4e-7, 4e-7, 1000))
        algo = DeltaRLE(tolerance=1e-6)
        compressed = algo.compress(data)
        max_error = np.max(np.abs(algo.decompress(compressed) - data))
        self.assertLessEqual(max_error, 1e-6 + 1e-12)
        self.assertLess(len(compressed), len(data) // 2)

    def test_delta_rle_zero_tolerance_is_lossless(self):
        rng = np.random.default_rng(0)
        for data in [np.arange(100, dtype=float), rng.normal(size=100)]:
            algo = DeltaRLE(tolerance=0)
            np.testing.assert_allclose(algo.decompress(algo.compress(data)), data,
                                       rtol=0, atol=1e-12)
        self.assertEqual(len(DeltaRLE(tolerance=0).compress(np.arange(100, dtype=float))), 1)

    def test_delta_rle_invalid_tolerance(self):
        with self.assertRaises(ValueError):
            DeltaRLE(tolerance=-1)

    def test_pca_compresses_univariate(self):
        """Windowed PCA must store less than the original series."""
        rng = np.random.default_rng(0)
        data = np.sin(np.arange(0, 100, 0.01)) + rng.normal(0, 0.1, 10000)
        result = TimeSeriesCompressor().benchmark_algorithm(data, PCACompression())
        self.assertGreater(result['Compression_Ratio'], 5)
        self.assertLess(result['MSE'], 0.02)

    def test_pca_multivariate(self):
        """2-D input is reduced across features."""
        rng = np.random.default_rng(0)
        base = rng.normal(size=200)
        data = np.column_stack([base, 2 * base + 1, -base])
        algo = PCACompression(n_components=0.99)
        compressed = algo.compress(data)
        self.assertEqual(compressed[0].shape, (200, 1))
        np.testing.assert_allclose(algo.decompress(compressed), data, atol=1e-10)

    def test_pca_short_series(self):
        """Series too short for PCA still round-trip."""
        for n in [1, 2, 3]:
            data = np.arange(n, dtype=float)
            algo = PCACompression()
            np.testing.assert_allclose(algo.decompress(algo.compress(data)), data, atol=1e-10)

    def test_pca_int_components_clamped(self):
        """An integer n_components larger than the matrix is clamped."""
        data = np.sin(np.arange(0, 10, 0.1))
        algo = PCACompression(n_components=50, window_size=10)
        np.testing.assert_allclose(algo.decompress(algo.compress(data)), data, atol=1e-10)

    def test_pca_invalid_window(self):
        with self.assertRaises(ValueError):
            PCACompression(window_size=0)

if __name__ == '__main__':
    unittest.main()
