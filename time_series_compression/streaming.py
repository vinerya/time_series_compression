"""Streaming compression implementations."""

import numpy as np
from typing import Optional

from .core import CompressedResult, StreamingCompressionAlgorithm, register_algorithm
from .serialization import pack_delta_rle_runs, unpack_delta_rle_runs


@register_algorithm
class StreamingDeltaRLE(StreamingCompressionAlgorithm):
    """Streaming delta + run-length encoding that processes data in chunks.

    Usage:
        algo = StreamingDeltaRLE(tolerance=1e-6)
        algo.partial_compress(chunk1)
        algo.partial_compress(chunk2)
        result = algo.finalize_compression()
        data = algo.decompress(result)
    """

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        self._runs: list = []
        self._last_value: Optional[float] = None
        self._current_start: float = 0.0
        self._current_delta: float = 0.0
        self._current_count: int = 0
        self._in_run: bool = False
        self._total_points: int = 0
        self._dtype: str = "float64"

    def reset(self) -> None:
        self._runs = []
        self._last_value = None
        self._current_start = 0.0
        self._current_delta = 0.0
        self._current_count = 0
        self._in_run = False
        self._total_points = 0
        self._dtype = "float64"

    def partial_compress(self, chunk: np.ndarray) -> Optional[bytes]:
        if not isinstance(chunk, np.ndarray) or chunk.size == 0:
            return None

        self._dtype = str(chunk.dtype)
        completed_runs = []

        for value in chunk.flat:
            value = float(value)
            self._total_points += 1

            if self._last_value is None:
                # Very first value
                self._current_start = value
                self._last_value = value
                self._in_run = False
                continue

            delta = value - self._last_value

            if not self._in_run:
                # Starting a new run with this delta
                self._current_delta = delta
                self._current_count = 1
                self._in_run = True
            elif abs(delta - self._current_delta) < self.tolerance:
                # Continues current run
                self._current_count += 1
            else:
                # Run broken — flush it and start a new run at last_value
                completed_runs.append(
                    (self._current_start, self._current_count, self._current_delta)
                )
                self._current_start = self._last_value
                self._current_delta = delta
                self._current_count = 1
                self._in_run = True

            self._last_value = value

        # Store completed runs
        self._runs.extend(completed_runs)

        if completed_runs:
            return pack_delta_rle_runs(completed_runs)
        return None

    def finalize_compression(self) -> CompressedResult:
        # Flush the last run
        if self._in_run:
            self._runs.append(
                (self._current_start, self._current_count, self._current_delta)
            )
        elif self._total_points == 1:
            # Single value
            self._runs.append((self._current_start, 0, 0.0))

        result = CompressedResult(
            algorithm="StreamingDeltaRLE",
            original_shape=(self._total_points,),
            original_dtype=self._dtype,
            params={"_constructor_args": {"tolerance": self.tolerance}},
            payload=pack_delta_rle_runs(self._runs),
        )
        self.reset()
        return result

    def compress(self, data: np.ndarray) -> CompressedResult:
        """Non-streaming batch compress (for compatibility)."""
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a numpy array")
        if data.size == 0:
            raise ValueError("Input data cannot be empty")
        self.reset()
        self.partial_compress(data)
        return self.finalize_compression()

    def decompress(self, result: CompressedResult) -> np.ndarray:
        self._validate_result(result)
        runs = unpack_delta_rle_runs(result.payload)
        values = []
        for idx, (start_val, count, delta) in enumerate(runs):
            segment = [start_val + i * delta for i in range(count + 1)]
            if idx < len(runs) - 1:
                values.extend(segment[:-1])
            else:
                values.extend(segment)
        return np.array(values, dtype=result.original_dtype)

    def __repr__(self):
        return f"StreamingDeltaRLE(tolerance={self.tolerance})"
