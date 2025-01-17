import unittest
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
        self.data = np.sin(self.time) + np.random.normal(0, 0.1, self.time.shape)

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

        self.assertEqual(compressed_data.shape[0], 10)
        self.assertEqual(self.data.shape, decompressed_data.shape)
        self.assertFalse(np.array_equal(self.data, compressed_data))
        np.testing.assert_allclose(self.data, decompressed_data, rtol=1e-1, atol=1e-1)

    def test_compress_decompress_sax(self):
        self.compressor.set_algorithm(SAX(segments=10, alphabet_size=5))
        compressed_data = self.compressor.compress(self.data)
        decompressed_data = self.compressor.decompress(compressed_data)

        self.assertEqual(compressed_data.shape[0], 10)
        self.assertEqual(self.data.shape, decompressed_data.shape)
        self.assertFalse(np.array_equal(self.data, compressed_data))
        np.testing.assert_allclose(self.data, decompressed_data, rtol=1e-1, atol=1e-1)

    def test_compress_decompress_dct(self):
        self.compressor.set_algorithm(DCT(keep_coeffs=10))
        compressed_data = self.compressor.compress(self.data)
        decompressed_data = self.compressor.decompress(compressed_data)

        self.assertEqual(compressed_data.shape[0], 10)
        self.assertEqual(self.data.shape, decompressed_data.shape)
        self.assertFalse(np.array_equal(self.data, compressed_data))
        np.testing.assert_allclose(self.data, decompressed_data, rtol=1e-1, atol=1e-1)

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

        self.assertIsInstance(compressed_data, list)
        self.assertEqual(self.data.shape, decompressed_data.shape)
        np.testing.assert_allclose(self.data, decompressed_data, rtol=1e-1, atol=1e-1)

    def test_compression_ratio(self):
        original_size = self.data.nbytes

        for algorithm in [DifferenceEncoding(), PAA(segments=10), SAX(segments=10, alphabet_size=5), 
                          DCT(keep_coeffs=10), RunLengthEncoding(), ZlibCompression(),
                          DiscreteWaveletTransform(wavelet='db4', level=3, threshold=0.1)]:
            self.compressor.set_algorithm(algorithm)
            compressed_data = self.compressor.compress(self.data)
            
            if isinstance(compressed_data, list):
                compressed_size = sum(arr.nbytes if isinstance(arr, np.ndarray) else len(str(arr)) for arr in compressed_data)
            elif isinstance(compressed_data, bytes):
                compressed_size = len(compressed_data)
            else:
                compressed_size = compressed_data.nbytes
            
            ratio = original_size / compressed_size
            self.assertGreater(ratio, 0.5, f"{algorithm.__class__.__name__} compression ratio is too low")

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
        self.data = np.sin(self.time) + np.random.normal(0, 0.1, self.time.shape)

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
        self.assertEqual(len(compressed_data), 3)  # data, components, scaler_params
        self.assertEqual(self.data.shape, decompressed_data.shape)
        np.testing.assert_allclose(self.data, decompressed_data, rtol=1e-1, atol=1e-1)

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
            'Algorithm', 'Compression_Ratio', 'MSE',
            'Compression_Time', 'Decompression_Time'
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

if __name__ == '__main__':
    unittest.main()
