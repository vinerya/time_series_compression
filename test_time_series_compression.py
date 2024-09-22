import unittest
import numpy as np
from time_series_compression import TimeSeriesCompressor, DifferenceEncoding, PAA, SAX, DCT, RunLengthEncoding, ZlibCompression, DiscreteWaveletTransform

class TestTimeSeriesCompressor(unittest.TestCase):
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

if __name__ == '__main__':
    unittest.main()