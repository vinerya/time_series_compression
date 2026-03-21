"""Stateless compression algorithm implementations.

Every algorithm's compress() returns a CompressedResult containing all metadata
needed for decompression. No per-compression state is stored on self.
"""

import zlib
import numpy as np
import pywt
from scipy.stats import norm
from scipy.fft import dct, idct
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as _PCA
from typing import Optional, Union

from .core import CompressionAlgorithm, CompressedResult, register_algorithm
from .serialization import (
    pack_numpy, unpack_numpy,
    pack_rle_runs, unpack_rle_runs,
    pack_delta_rle_runs, unpack_delta_rle_runs,
    pack_wavelet_coeffs, unpack_wavelet_coeffs,
)


@register_algorithm
class DifferenceEncoding(CompressionAlgorithm):
    """Lossless delta encoding — stores differences between consecutive values."""

    def compress(self, data: np.ndarray) -> CompressedResult:
        self._validate_input(data)
        compressed = np.empty_like(data)
        compressed[0] = data[0]
        compressed[1:] = np.diff(data)
        return CompressedResult(
            algorithm="DifferenceEncoding",
            original_shape=data.shape,
            original_dtype=str(data.dtype),
            params={"_constructor_args": {}},
            payload=pack_numpy(compressed),
        )

    def decompress(self, result: CompressedResult) -> np.ndarray:
        self._validate_result(result)
        arr = unpack_numpy(result.payload, result.original_dtype, result.original_shape)
        return np.cumsum(arr)

    def __repr__(self):
        return "DifferenceEncoding()"


@register_algorithm
class PAA(CompressionAlgorithm):
    """Piecewise Aggregate Approximation — lossy segment averaging."""

    def __init__(self, segments: int = 10):
        if segments <= 0:
            raise ValueError("segments must be a positive integer")
        self.segments = segments

    def compress(self, data: np.ndarray) -> CompressedResult:
        self._validate_input(data)
        chunks = np.array_split(data, self.segments)
        means = np.array([c.mean() for c in chunks], dtype=np.float64)
        chunk_sizes = [len(c) for c in chunks]
        return CompressedResult(
            algorithm="PAA",
            original_shape=data.shape,
            original_dtype=str(data.dtype),
            params={
                "_constructor_args": {"segments": self.segments},
                "chunk_sizes": chunk_sizes,
            },
            payload=pack_numpy(means),
        )

    def decompress(self, result: CompressedResult) -> np.ndarray:
        self._validate_result(result)
        means = unpack_numpy(result.payload, "float64", (len(result.params["chunk_sizes"]),))
        return np.repeat(means, result.params["chunk_sizes"]).astype(result.original_dtype)

    def __repr__(self):
        return f"PAA(segments={self.segments})"


@register_algorithm
class SAX(CompressionAlgorithm):
    """Symbolic Aggregate approXimation — lossy symbolic compression."""

    def __init__(self, segments: int = 10, alphabet_size: int = 5):
        if segments <= 0:
            raise ValueError("segments must be a positive integer")
        if alphabet_size <= 1:
            raise ValueError("alphabet_size must be greater than 1")
        self.segments = segments
        self.alphabet_size = alphabet_size

    def compress(self, data: np.ndarray) -> CompressedResult:
        self._validate_input(data)
        mean = float(np.mean(data))
        std = float(np.std(data))
        breakpoints = norm.ppf(np.linspace(0, 1, self.alphabet_size + 1)[1:-1])

        normalized = (data - mean) / std if std > 0 else np.zeros_like(data)

        chunks = np.array_split(normalized, self.segments)
        paa_data = np.array([c.mean() for c in chunks], dtype=np.float64)
        chunk_sizes = [len(c) for c in chunks]

        symbols = np.digitize(paa_data, breakpoints).astype(np.int32)

        return CompressedResult(
            algorithm="SAX",
            original_shape=data.shape,
            original_dtype=str(data.dtype),
            params={
                "_constructor_args": {"segments": self.segments,
                                      "alphabet_size": self.alphabet_size},
                "breakpoints": breakpoints,
                "original_mean": mean,
                "original_std": std,
                "chunk_sizes": chunk_sizes,
            },
            payload=pack_numpy(symbols),
        )

    def decompress(self, result: CompressedResult) -> np.ndarray:
        self._validate_result(result)
        p = result.params
        breakpoints = np.asarray(p["breakpoints"])
        symbols = unpack_numpy(result.payload, "int32", (len(p["chunk_sizes"]),))
        # Clip indices: digitize returns 0..len(breakpoints), map to valid breakpoint indices
        indices = np.clip(symbols - 1, 0, len(breakpoints) - 1)
        paa_data = breakpoints[indices]
        reconstructed = np.repeat(paa_data, p["chunk_sizes"])
        return (reconstructed * p["original_std"] + p["original_mean"]).astype(result.original_dtype)

    def __repr__(self):
        return f"SAX(segments={self.segments}, alphabet_size={self.alphabet_size})"


@register_algorithm
class DCT(CompressionAlgorithm):
    """Discrete Cosine Transform — lossy frequency-domain compression."""

    def __init__(self, keep_coeffs: int = 20):
        if keep_coeffs <= 0:
            raise ValueError("keep_coeffs must be a positive integer")
        self.keep_coeffs = keep_coeffs

    def compress(self, data: np.ndarray) -> CompressedResult:
        self._validate_input(data)
        coeffs = dct(data.astype(np.float64))
        kept = coeffs[:self.keep_coeffs].copy()
        return CompressedResult(
            algorithm="DCT",
            original_shape=data.shape,
            original_dtype=str(data.dtype),
            params={"_constructor_args": {"keep_coeffs": self.keep_coeffs}},
            payload=pack_numpy(kept),
        )

    def decompress(self, result: CompressedResult) -> np.ndarray:
        self._validate_result(result)
        n_coeffs = result.params.get("_constructor_args", {}).get("keep_coeffs", 0)
        kept = unpack_numpy(result.payload, "float64", (n_coeffs,))
        full = np.zeros(result.original_shape, dtype=np.float64)
        full[:len(kept)] = kept
        return idct(full, n=result.original_shape[0]).astype(result.original_dtype)

    def __repr__(self):
        return f"DCT(keep_coeffs={self.keep_coeffs})"


@register_algorithm
class RunLengthEncoding(CompressionAlgorithm):
    """Lossless run-length encoding — compresses consecutive identical values."""

    def compress(self, data: np.ndarray) -> CompressedResult:
        self._validate_input(data)
        runs = []
        count = 1
        for i in range(1, len(data)):
            if data[i] == data[i - 1]:
                count += 1
            else:
                runs.append((data[i - 1], count))
                count = 1
        runs.append((data[-1], count))
        return CompressedResult(
            algorithm="RunLengthEncoding",
            original_shape=data.shape,
            original_dtype=str(data.dtype),
            params={"_constructor_args": {}},
            payload=pack_rle_runs(runs),
        )

    def decompress(self, result: CompressedResult) -> np.ndarray:
        self._validate_result(result)
        runs = unpack_rle_runs(result.payload)
        values = []
        for value, count in runs:
            values.extend([value] * count)
        return np.array(values, dtype=result.original_dtype).reshape(result.original_shape)

    def __repr__(self):
        return "RunLengthEncoding()"


@register_algorithm
class ZlibCompression(CompressionAlgorithm):
    """Lossless general-purpose compression using zlib."""

    def __init__(self, level: int = 6):
        self.level = level

    def compress(self, data: np.ndarray) -> CompressedResult:
        self._validate_input(data)
        return CompressedResult(
            algorithm="ZlibCompression",
            original_shape=data.shape,
            original_dtype=str(data.dtype),
            params={"_constructor_args": {"level": self.level}},
            payload=zlib.compress(data.tobytes(), self.level),
        )

    def decompress(self, result: CompressedResult) -> np.ndarray:
        self._validate_result(result)
        raw = zlib.decompress(result.payload)
        return np.frombuffer(raw, dtype=result.original_dtype).reshape(
            result.original_shape
        ).copy()

    def __repr__(self):
        return f"ZlibCompression(level={self.level})"


@register_algorithm
class DiscreteWaveletTransform(CompressionAlgorithm):
    """Lossy wavelet compression with configurable thresholding."""

    def __init__(self, wavelet: str = 'db4', level: Optional[int] = None,
                 threshold: float = 0.1):
        self.wavelet = wavelet
        self.level = level
        self.threshold = threshold

    def compress(self, data: np.ndarray) -> CompressedResult:
        self._validate_input(data)
        coeffs = pywt.wavedec(data.astype(np.float64), self.wavelet, level=self.level)
        for i in range(1, len(coeffs)):
            max_abs = np.max(np.abs(coeffs[i]))
            if max_abs > 0:
                coeffs[i] = pywt.threshold(coeffs[i], self.threshold * max_abs)
        payload, coeff_lengths = pack_wavelet_coeffs(coeffs)
        return CompressedResult(
            algorithm="DiscreteWaveletTransform",
            original_shape=data.shape,
            original_dtype=str(data.dtype),
            params={
                "_constructor_args": {"wavelet": self.wavelet, "level": self.level,
                                      "threshold": self.threshold},
                "coeff_lengths": coeff_lengths,
            },
            payload=payload,
        )

    def decompress(self, result: CompressedResult) -> np.ndarray:
        self._validate_result(result)
        p = result.params
        constructor = p.get("_constructor_args", {})
        coeffs = unpack_wavelet_coeffs(result.payload, p["coeff_lengths"])
        wavelet = constructor.get("wavelet", self.wavelet)
        reconstructed = pywt.waverec(coeffs, wavelet)
        return reconstructed[:result.original_shape[0]].astype(result.original_dtype)

    def __repr__(self):
        return (f"DiscreteWaveletTransform(wavelet='{self.wavelet}', "
                f"level={self.level}, threshold={self.threshold})")


@register_algorithm
class DeltaRLE(CompressionAlgorithm):
    """Hybrid delta + run-length encoding. Effective for constant-rate-of-change data."""

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance

    def compress(self, data: np.ndarray) -> CompressedResult:
        self._validate_input(data)
        runs = self._compute_runs(data)
        return CompressedResult(
            algorithm="DeltaRLE",
            original_shape=data.shape,
            original_dtype=str(data.dtype),
            params={"_constructor_args": {"tolerance": self.tolerance}},
            payload=pack_delta_rle_runs(runs),
        )

    def _compute_runs(self, data: np.ndarray) -> list:
        deltas = np.diff(data)
        if len(deltas) == 0:
            return [(float(data[0]), 0, 0.0)]

        runs = []
        start_val = float(data[0])
        current_delta = float(deltas[0])
        count = 1

        for i in range(1, len(deltas)):
            if abs(deltas[i] - current_delta) < self.tolerance:
                count += 1
            else:
                runs.append((start_val, count, current_delta))
                start_val = float(data[i])
                current_delta = float(deltas[i])
                count = 1
        runs.append((start_val, count, current_delta))
        return runs

    def decompress(self, result: CompressedResult) -> np.ndarray:
        self._validate_result(result)
        runs = unpack_delta_rle_runs(result.payload)
        return self._reconstruct(runs, result.original_dtype)

    @staticmethod
    def _reconstruct(runs: list, dtype: str = "float64") -> np.ndarray:
        values = []
        for idx, (start_val, count, delta) in enumerate(runs):
            segment = [start_val + i * delta for i in range(count + 1)]
            if idx < len(runs) - 1:
                values.extend(segment[:-1])
            else:
                values.extend(segment)
        return np.array(values, dtype=dtype)

    def __repr__(self):
        return f"DeltaRLE(tolerance={self.tolerance})"


@register_algorithm
class PCACompression(CompressionAlgorithm):
    """PCA-based compression for multivariate (2D) time series.

    Input must be 2D: shape (n_timesteps, n_features) with n_features >= 2.
    """

    def __init__(self, n_components: Union[int, float] = 0.95):
        self.n_components = n_components

    def compress(self, data: np.ndarray) -> CompressedResult:
        self._validate_input(data)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(
                "PCACompression requires 2D data with shape (n_timesteps, n_features) "
                "where n_features >= 2. For 1D time series, use DCT or DWT instead."
            )

        scaler = StandardScaler()
        scaled = scaler.fit_transform(data.astype(np.float64))

        pca = _PCA(n_components=self.n_components)
        transformed = pca.fit_transform(scaled)

        return CompressedResult(
            algorithm="PCACompression",
            original_shape=data.shape,
            original_dtype=str(data.dtype),
            params={
                "_constructor_args": {"n_components": self.n_components},
                "components": pca.components_,
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "n_components_actual": pca.n_components_,
            },
            payload=pack_numpy(transformed),
        )

    def decompress(self, result: CompressedResult) -> np.ndarray:
        self._validate_result(result)
        p = result.params
        components = np.asarray(p["components"])
        mean = np.asarray(p["scaler_mean"])
        scale = np.asarray(p["scaler_scale"])

        n_actual = p["n_components_actual"]
        transformed = unpack_numpy(
            result.payload, "float64",
            (result.original_shape[0], n_actual)
        )
        reconstructed = transformed @ components
        reconstructed = reconstructed * scale + mean
        return reconstructed.astype(result.original_dtype)

    def __repr__(self):
        return f"PCACompression(n_components={self.n_components})"
