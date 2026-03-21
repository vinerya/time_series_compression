"""Time Series Compression — a production-ready framework for compressing time series data.

All algorithms return CompressedResult objects that can be serialized to/from bytes
for storage and transmission. Algorithms are stateless and thread-safe.
"""

from .core import (
    CompressedResult,
    CompressionAlgorithm,
    StreamingCompressionAlgorithm,
    TimeSeriesCompressor,
)
from .algorithms import (
    DifferenceEncoding,
    PAA,
    SAX,
    DCT,
    RunLengthEncoding,
    ZlibCompression,
    DiscreteWaveletTransform,
    DeltaRLE,
    PCACompression,
)
from .streaming import StreamingDeltaRLE

__all__ = [
    "CompressedResult",
    "CompressionAlgorithm",
    "StreamingCompressionAlgorithm",
    "TimeSeriesCompressor",
    "DifferenceEncoding",
    "PAA",
    "SAX",
    "DCT",
    "RunLengthEncoding",
    "ZlibCompression",
    "DiscreteWaveletTransform",
    "DeltaRLE",
    "PCACompression",
    "StreamingDeltaRLE",
]

__version__ = "2.0.0"
