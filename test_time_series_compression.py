import unittest
import numpy as np
import pandas as pd
from time_series_compression import (
    CompressedResult,
    TimeSeriesCompressor,
    CompressionAlgorithm,
    StreamingCompressionAlgorithm,
    DifferenceEncoding,
    PAA,
    SAX,
    DCT,
    RunLengthEncoding,
    ZlibCompression,
    DiscreteWaveletTransform,
    DeltaRLE,
    PCACompression,
    StreamingDeltaRLE,
)


class TestCompressedResult(unittest.TestCase):
    """Tests for CompressedResult serialization round-trips."""

    def test_to_bytes_from_bytes_roundtrip(self):
        cr = CompressedResult(
            algorithm="test",
            original_shape=(100,),
            original_dtype="float64",
            params={"foo": 42, "bar": [1, 2, 3]},
            payload=b"\x00\x01\x02\x03",
        )
        restored = CompressedResult.from_bytes(cr.to_bytes())
        self.assertEqual(restored.algorithm, cr.algorithm)
        self.assertEqual(restored.original_shape, cr.original_shape)
        self.assertEqual(restored.original_dtype, cr.original_dtype)
        self.assertEqual(restored.params["foo"], 42)
        self.assertEqual(restored.payload, cr.payload)

    def test_invalid_magic(self):
        with self.assertRaises(ValueError):
            CompressedResult.from_bytes(b"NOPE" + b"\x00" * 20)

    def test_compression_ratio(self):
        cr = CompressedResult(
            algorithm="test",
            original_shape=(1000,),
            original_dtype="float64",
            params={},
            payload=b"\x00" * 100,
        )
        self.assertGreater(cr.compression_ratio, 1.0)

    def test_numpy_array_in_params(self):
        """Params containing numpy arrays should survive serialization."""
        cr = CompressedResult(
            algorithm="test",
            original_shape=(10,),
            original_dtype="float64",
            params={"breakpoints": np.array([0.1, 0.5, 0.9])},
            payload=b"\x00",
        )
        restored = CompressedResult.from_bytes(cr.to_bytes())
        np.testing.assert_allclose(
            restored.params["breakpoints"], [0.1, 0.5, 0.9]
        )


class TestLosslessAlgorithms(unittest.TestCase):
    """Lossless algorithms must reconstruct data exactly."""

    def setUp(self):
        rng = np.random.default_rng(42)
        self.data = np.sin(np.arange(0, 10, 0.1)) + rng.normal(0, 0.1, 100)

    def _roundtrip(self, algo):
        result = algo.compress(self.data)
        self.assertIsInstance(result, CompressedResult)
        decompressed = algo.decompress(result)
        self.assertEqual(decompressed.shape, self.data.shape)
        return decompressed

    def test_difference_encoding(self):
        decompressed = self._roundtrip(DifferenceEncoding())
        np.testing.assert_allclose(self.data, decompressed, atol=1e-10)

    def test_run_length_encoding(self):
        decompressed = self._roundtrip(RunLengthEncoding())
        np.testing.assert_allclose(self.data, decompressed, atol=1e-10)

    def test_zlib(self):
        decompressed = self._roundtrip(ZlibCompression())
        np.testing.assert_array_equal(self.data, decompressed)

    def test_zlib_preserves_dtype(self):
        for dtype in [np.float32, np.float64, np.int32]:
            data = np.array([1, 2, 3, 4, 5], dtype=dtype)
            result = ZlibCompression().compress(data)
            decompressed = ZlibCompression().decompress(result)
            self.assertEqual(decompressed.dtype, dtype)
            np.testing.assert_array_equal(data, decompressed)

    def test_delta_rle(self):
        decompressed = self._roundtrip(DeltaRLE())
        np.testing.assert_allclose(self.data, decompressed, atol=1e-6)

    def test_delta_rle_constant_data(self):
        data = np.ones(100)
        result = DeltaRLE().compress(data)
        decompressed = DeltaRLE().decompress(result)
        np.testing.assert_array_equal(data, decompressed)
        # Should compress to a single run
        self.assertLess(len(result.payload), data.nbytes)

    def test_delta_rle_single_value(self):
        data = np.array([42.0])
        result = DeltaRLE().compress(data)
        decompressed = DeltaRLE().decompress(result)
        np.testing.assert_array_equal(data, decompressed)


class TestLossyAlgorithms(unittest.TestCase):
    """Lossy algorithms must reconstruct with bounded error."""

    def setUp(self):
        rng = np.random.default_rng(42)
        self.data = np.sin(np.arange(0, 10, 0.1)) + rng.normal(0, 0.1, 100)

    def _roundtrip_mse(self, algo, max_mse):
        result = algo.compress(self.data)
        self.assertIsInstance(result, CompressedResult)
        decompressed = algo.decompress(result)
        self.assertEqual(decompressed.shape, self.data.shape)
        mse = np.mean((self.data - decompressed) ** 2)
        self.assertLess(mse, max_mse, f"{algo} MSE {mse} exceeds {max_mse}")
        return result

    def test_paa(self):
        self._roundtrip_mse(PAA(segments=10), max_mse=0.5)

    def test_paa_non_divisible_length(self):
        data = np.arange(17, dtype=float)
        result = PAA(segments=5).compress(data)
        decompressed = PAA(segments=5).decompress(result)
        self.assertEqual(len(decompressed), 17)

    def test_sax(self):
        self._roundtrip_mse(SAX(segments=10, alphabet_size=5), max_mse=1.0)

    def test_dct(self):
        self._roundtrip_mse(DCT(keep_coeffs=10), max_mse=0.5)

    def test_dwt(self):
        self._roundtrip_mse(
            DiscreteWaveletTransform(wavelet='db4', level=3, threshold=0.1),
            max_mse=0.1,
        )

    def test_pca_2d(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((50, 5))
        algo = PCACompression(n_components=0.95)
        result = algo.compress(data)
        decompressed = algo.decompress(result)
        self.assertEqual(decompressed.shape, data.shape)
        mse = np.mean((data - decompressed) ** 2)
        self.assertLess(mse, 1.0)

    def test_pca_rejects_1d(self):
        with self.assertRaises(ValueError):
            PCACompression().compress(np.array([1.0, 2.0, 3.0]))


class TestByteSerializationRoundtrip(unittest.TestCase):
    """Every algorithm's output must survive to_bytes/from_bytes."""

    def setUp(self):
        rng = np.random.default_rng(42)
        self.data = np.sin(np.arange(0, 10, 0.1)) + rng.normal(0, 0.1, 100)

    def _test_algo_bytes_roundtrip(self, algo):
        result = algo.compress(self.data)
        serialized = result.to_bytes()
        self.assertIsInstance(serialized, bytes)
        restored = CompressedResult.from_bytes(serialized)
        decompressed = algo.decompress(restored)
        # Must match the non-serialized decompression
        original_decompressed = algo.decompress(result)
        np.testing.assert_array_equal(decompressed, original_decompressed)

    def test_difference_encoding(self):
        self._test_algo_bytes_roundtrip(DifferenceEncoding())

    def test_paa(self):
        self._test_algo_bytes_roundtrip(PAA(segments=10))

    def test_sax(self):
        self._test_algo_bytes_roundtrip(SAX(segments=10, alphabet_size=5))

    def test_dct(self):
        self._test_algo_bytes_roundtrip(DCT(keep_coeffs=10))

    def test_rle(self):
        self._test_algo_bytes_roundtrip(RunLengthEncoding())

    def test_zlib(self):
        self._test_algo_bytes_roundtrip(ZlibCompression())

    def test_dwt(self):
        self._test_algo_bytes_roundtrip(
            DiscreteWaveletTransform(wavelet='db4', level=3, threshold=0.1)
        )

    def test_delta_rle(self):
        self._test_algo_bytes_roundtrip(DeltaRLE())

    def test_pca(self):
        rng = np.random.default_rng(42)
        data_2d = rng.standard_normal((50, 5))
        algo = PCACompression(n_components=0.95)
        result = algo.compress(data_2d)
        serialized = result.to_bytes()
        restored = CompressedResult.from_bytes(serialized)
        decompressed = algo.decompress(restored)
        original_decompressed = algo.decompress(result)
        np.testing.assert_allclose(decompressed, original_decompressed, atol=1e-10)


class TestDecompressFromBytes(unittest.TestCase):
    """Test auto-detection decompression from raw bytes."""

    def setUp(self):
        rng = np.random.default_rng(42)
        self.data = np.sin(np.arange(0, 10, 0.1)) + rng.normal(0, 0.1, 100)

    def test_auto_decompress_difference_encoding(self):
        raw = TimeSeriesCompressor(DifferenceEncoding()).compress_to_bytes(self.data)
        restored = TimeSeriesCompressor.decompress_from_bytes(raw)
        np.testing.assert_allclose(self.data, restored, atol=1e-10)

    def test_auto_decompress_zlib(self):
        raw = TimeSeriesCompressor(ZlibCompression()).compress_to_bytes(self.data)
        restored = TimeSeriesCompressor.decompress_from_bytes(raw)
        np.testing.assert_array_equal(self.data, restored)

    def test_auto_decompress_delta_rle(self):
        raw = TimeSeriesCompressor(DeltaRLE()).compress_to_bytes(self.data)
        restored = TimeSeriesCompressor.decompress_from_bytes(raw)
        np.testing.assert_allclose(self.data, restored, atol=1e-6)


class TestStatelessness(unittest.TestCase):
    """Algorithms must be reusable across multiple datasets without interference."""

    def test_same_instance_two_datasets(self):
        rng = np.random.default_rng(42)
        data1 = rng.standard_normal(100)
        data2 = rng.standard_normal(50)

        algo = DeltaRLE()
        r1 = algo.compress(data1)
        r2 = algo.compress(data2)

        d1 = algo.decompress(r1)
        d2 = algo.decompress(r2)

        np.testing.assert_allclose(data1, d1, atol=1e-6)
        np.testing.assert_allclose(data2, d2, atol=1e-6)

    def test_stateless_zlib(self):
        data1 = np.array([1.0, 2.0, 3.0])
        data2 = np.arange(1000, dtype=np.float32)

        algo = ZlibCompression()
        r1 = algo.compress(data1)
        r2 = algo.compress(data2)

        np.testing.assert_array_equal(algo.decompress(r1), data1)
        np.testing.assert_array_equal(algo.decompress(r2), data2)


class TestTimeSeriesCompressor(unittest.TestCase):
    """Test the main compressor orchestrator."""

    def setUp(self):
        self.compressor = TimeSeriesCompressor()
        rng = np.random.default_rng(42)
        self.data = np.sin(np.arange(0, 10, 0.1)) + rng.normal(0, 0.1, 100)

    def test_default_algorithm(self):
        self.assertIsInstance(self.compressor.algorithm, DifferenceEncoding)

    def test_set_algorithm(self):
        algo = PAA(segments=10)
        self.compressor.set_algorithm(algo)
        self.assertIs(self.compressor.algorithm, algo)

    def test_invalid_algorithm(self):
        with self.assertRaises(TypeError):
            self.compressor.set_algorithm("not an algorithm")

    def test_empty_data(self):
        with self.assertRaises(ValueError):
            self.compressor.compress(np.array([]))

    def test_non_array_input(self):
        with self.assertRaises(TypeError):
            self.compressor.compress([1, 2, 3])

    def test_compress_to_bytes_and_back(self):
        raw = self.compressor.compress_to_bytes(self.data)
        self.assertIsInstance(raw, bytes)
        restored = TimeSeriesCompressor.decompress_from_bytes(raw)
        np.testing.assert_allclose(self.data, restored, atol=1e-10)

    def test_benchmark(self):
        algorithms = [DeltaRLE(), DifferenceEncoding()]
        results = self.compressor.benchmark_all(self.data, algorithms)
        self.assertIsInstance(results, pd.DataFrame)
        self.assertEqual(len(results), 2)
        for col in ['Algorithm', 'Compression_Ratio', 'MSE', 'Max_Error',
                     'SNR_dB', 'Compression_Time', 'Decompression_Time',
                     'Compressed_Bytes']:
            self.assertIn(col, results.columns)

    def test_auto_select(self):
        algorithms = [DeltaRLE(), DifferenceEncoding(), PAA(segments=10)]
        for priority in ['size', 'speed', 'accuracy', 'balanced']:
            best = self.compressor.auto_select_algorithm(
                self.data, algorithms, priority=priority
            )
            self.assertIsInstance(best, CompressionAlgorithm)

    def test_auto_select_single_algorithm(self):
        best = self.compressor.auto_select_algorithm(
            self.data, [DifferenceEncoding()], priority='balanced'
        )
        self.assertIsInstance(best, DifferenceEncoding)

    def test_invalid_priority(self):
        with self.assertRaises(ValueError):
            self.compressor.auto_select_algorithm(
                self.data, [DeltaRLE()], priority='invalid'
            )

    def test_parallel_compression(self):
        self.compressor.set_algorithm(DeltaRLE())
        results = self.compressor.compress_parallel(self.data, chunk_size=10)
        self.assertIsInstance(results, list)
        decompressed = self.compressor.decompress_parallel(results)
        np.testing.assert_allclose(self.data, decompressed, atol=1e-6)


class TestStreaming(unittest.TestCase):
    """Test streaming compression."""

    def setUp(self):
        rng = np.random.default_rng(42)
        self.data = np.sin(np.arange(0, 10, 0.1)) + rng.normal(0, 0.1, 100)

    def test_streaming_basic(self):
        algo = StreamingDeltaRLE(tolerance=1e-6)
        algo.partial_compress(self.data[:30])
        algo.partial_compress(self.data[30:70])
        algo.partial_compress(self.data[70:])
        result = algo.finalize_compression()

        self.assertIsInstance(result, CompressedResult)
        self.assertEqual(result.original_shape, (100,))

        decompressed = algo.decompress(result)
        np.testing.assert_allclose(self.data, decompressed, atol=1e-6)

    def test_streaming_matches_batch(self):
        algo_batch = DeltaRLE(tolerance=1e-6)
        batch_result = algo_batch.compress(self.data)
        batch_decompressed = algo_batch.decompress(batch_result)

        algo_stream = StreamingDeltaRLE(tolerance=1e-6)
        algo_stream.partial_compress(self.data[:50])
        algo_stream.partial_compress(self.data[50:])
        stream_result = algo_stream.finalize_compression()
        stream_decompressed = algo_stream.decompress(stream_result)

        np.testing.assert_allclose(
            batch_decompressed, stream_decompressed, atol=1e-6
        )

    def test_streaming_single_value(self):
        algo = StreamingDeltaRLE()
        algo.partial_compress(np.array([5.0]))
        result = algo.finalize_compression()
        decompressed = algo.decompress(result)
        np.testing.assert_array_equal(decompressed, [5.0])

    def test_streaming_reset(self):
        algo = StreamingDeltaRLE()
        algo.partial_compress(np.array([1.0, 2.0, 3.0]))
        algo.reset()
        algo.partial_compress(np.array([10.0, 20.0]))
        result = algo.finalize_compression()
        decompressed = algo.decompress(result)
        np.testing.assert_allclose(decompressed, [10.0, 20.0], atol=1e-10)

    def test_streaming_bytes_roundtrip(self):
        algo = StreamingDeltaRLE()
        algo.partial_compress(self.data)
        result = algo.finalize_compression()
        raw = result.to_bytes()
        restored = CompressedResult.from_bytes(raw)
        decompressed = algo.decompress(restored)
        np.testing.assert_allclose(self.data, decompressed, atol=1e-6)

    def test_streaming_abstract_enforcement(self):
        with self.assertRaises(TypeError):
            class Incomplete(StreamingCompressionAlgorithm):
                pass
            Incomplete()


class TestParameterValidation(unittest.TestCase):
    """Algorithms should reject invalid constructor parameters."""

    def test_paa_invalid_segments(self):
        with self.assertRaises(ValueError):
            PAA(segments=0)
        with self.assertRaises(ValueError):
            PAA(segments=-1)

    def test_sax_invalid_params(self):
        with self.assertRaises(ValueError):
            SAX(segments=0, alphabet_size=5)
        with self.assertRaises(ValueError):
            SAX(segments=10, alphabet_size=1)

    def test_dct_invalid_keep_coeffs(self):
        with self.assertRaises(ValueError):
            DCT(keep_coeffs=0)


class TestReprMethods(unittest.TestCase):
    def test_all_algorithms_have_repr(self):
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
            StreamingDeltaRLE(tolerance=1e-6),
        ]
        for algo in algos:
            self.assertIn(algo.__class__.__name__, repr(algo))


class TestActualCompression(unittest.TestCase):
    """Verify that algorithms actually reduce data size for suitable inputs."""

    def test_zlib_compresses_smooth_data(self):
        data = np.sin(np.linspace(0, 10, 10000))
        result = ZlibCompression().compress(data)
        self.assertLess(len(result.to_bytes()), data.nbytes)

    def test_delta_rle_compresses_constant(self):
        data = np.ones(10000)
        result = DeltaRLE().compress(data)
        self.assertLess(len(result.to_bytes()), data.nbytes)

    def test_dct_compresses(self):
        data = np.sin(np.linspace(0, 10, 1000))
        result = DCT(keep_coeffs=20).compress(data)
        self.assertLess(len(result.to_bytes()), data.nbytes)

    def test_paa_compresses(self):
        data = np.sin(np.linspace(0, 10, 1000))
        result = PAA(segments=50).compress(data)
        self.assertLess(len(result.to_bytes()), data.nbytes)


if __name__ == '__main__':
    unittest.main()
