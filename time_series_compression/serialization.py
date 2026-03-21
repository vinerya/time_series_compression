"""Binary serialization for CompressedResult.

Format:
  Bytes 0-3:   Magic b"TSCP"
  Bytes 4-5:   Version (uint16 big-endian, currently 1)
  Bytes 6-9:   Header length (uint32 big-endian)
  Bytes 10-N:  JSON-encoded header dict (algorithm, shape, dtype, params)
  Bytes N+1-:  Raw payload bytes

Uses JSON for the header (no extra dependencies) and raw bytes for payload.
"""

import json
import struct
import numpy as np

MAGIC = b"TSCP"
VERSION = 1
HEADER_STRUCT = struct.Struct(">HI")  # version (uint16) + header_len (uint32)


def serialize_compressed_result(cr) -> bytes:
    """Serialize a CompressedResult to bytes."""
    header = {
        "algorithm": cr.algorithm,
        "original_shape": list(cr.original_shape),
        "original_dtype": cr.original_dtype,
        "params": _params_to_json_safe(cr.params),
    }
    header_bytes = json.dumps(header, separators=(',', ':')).encode('utf-8')
    buf = bytearray()
    buf.extend(MAGIC)
    buf.extend(HEADER_STRUCT.pack(VERSION, len(header_bytes)))
    buf.extend(header_bytes)
    buf.extend(cr.payload)
    return bytes(buf)


def deserialize_compressed_result(data: bytes):
    """Deserialize bytes to a CompressedResult."""
    from .core import CompressedResult

    if len(data) < 10:
        raise ValueError("Data too short to be a valid TSCP stream")
    if data[:4] != MAGIC:
        raise ValueError(f"Invalid magic bytes: expected {MAGIC!r}, got {data[:4]!r}")

    version, header_len = HEADER_STRUCT.unpack_from(data, 4)
    if version != VERSION:
        raise ValueError(f"Unsupported version: {version}")

    header_start = 10
    header_end = header_start + header_len
    if len(data) < header_end:
        raise ValueError("Data truncated: header incomplete")

    header = json.loads(data[header_start:header_end].decode('utf-8'))
    payload = data[header_end:]

    return CompressedResult(
        algorithm=header["algorithm"],
        original_shape=tuple(header["original_shape"]),
        original_dtype=header["original_dtype"],
        params=_params_from_json(header["params"]),
        payload=payload,
    )


# --- Packing helpers for algorithm payloads ---

_RLE_PAIR = struct.Struct("<dI")    # (float64, uint32) = 12 bytes per run
_DELTA_RLE_TRIPLE = struct.Struct("<dId")  # (float64, uint32, float64) = 20 bytes


def pack_numpy(arr: np.ndarray) -> bytes:
    """Pack a numpy array to raw bytes (contiguous, C-order)."""
    return np.ascontiguousarray(arr).tobytes()


def unpack_numpy(data: bytes, dtype: str, shape: tuple) -> np.ndarray:
    """Unpack raw bytes to a numpy array."""
    return np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape).copy()


def pack_rle_runs(runs: list) -> bytes:
    """Pack RLE (value, count) pairs to bytes."""
    buf = bytearray(len(runs) * _RLE_PAIR.size)
    for i, (value, count) in enumerate(runs):
        _RLE_PAIR.pack_into(buf, i * _RLE_PAIR.size, float(value), int(count))
    return bytes(buf)


def unpack_rle_runs(data: bytes) -> list:
    """Unpack bytes to RLE (value, count) pairs."""
    n = len(data) // _RLE_PAIR.size
    runs = []
    for i in range(n):
        value, count = _RLE_PAIR.unpack_from(data, i * _RLE_PAIR.size)
        runs.append((value, count))
    return runs


def pack_delta_rle_runs(runs: list) -> bytes:
    """Pack DeltaRLE (start_val, count, delta) triples to bytes."""
    buf = bytearray(len(runs) * _DELTA_RLE_TRIPLE.size)
    for i, (start_val, count, delta) in enumerate(runs):
        _DELTA_RLE_TRIPLE.pack_into(
            buf, i * _DELTA_RLE_TRIPLE.size,
            float(start_val), int(count), float(delta)
        )
    return bytes(buf)


def unpack_delta_rle_runs(data: bytes) -> list:
    """Unpack bytes to DeltaRLE (start_val, count, delta) triples."""
    n = len(data) // _DELTA_RLE_TRIPLE.size
    runs = []
    for i in range(n):
        start_val, count, delta = _DELTA_RLE_TRIPLE.unpack_from(
            data, i * _DELTA_RLE_TRIPLE.size
        )
        runs.append((start_val, count, delta))
    return runs


def pack_wavelet_coeffs(coeffs: list) -> tuple:
    """Pack wavelet coefficients into bytes + length metadata.

    Returns (payload_bytes, coeff_lengths) where coeff_lengths is
    a list of ints giving the length of each coefficient array.
    """
    coeff_lengths = [len(c) for c in coeffs]
    flat = np.concatenate([np.asarray(c, dtype=np.float64) for c in coeffs])
    return flat.tobytes(), coeff_lengths


def unpack_wavelet_coeffs(data: bytes, coeff_lengths: list) -> list:
    """Unpack bytes into list of wavelet coefficient arrays."""
    flat = np.frombuffer(data, dtype=np.float64)
    coeffs = []
    offset = 0
    for length in coeff_lengths:
        coeffs.append(flat[offset:offset + length].copy())
        offset += length
    return coeffs


# --- JSON-safe param conversion ---

def _params_to_json_safe(params: dict) -> dict:
    """Convert params dict values to JSON-serializable types."""
    result = {}
    for key, value in params.items():
        if isinstance(value, np.ndarray):
            result[key] = {"__ndarray__": True, "data": value.tolist(),
                           "dtype": str(value.dtype)}
        elif isinstance(value, np.integer):
            result[key] = int(value)
        elif isinstance(value, np.floating):
            result[key] = float(value)
        elif isinstance(value, (list, tuple)):
            result[key] = [float(v) if isinstance(v, np.floating) else
                           int(v) if isinstance(v, np.integer) else v
                           for v in value]
        else:
            result[key] = value
    return result


def _params_from_json(params: dict) -> dict:
    """Restore params dict from JSON-deserialized form."""
    result = {}
    for key, value in params.items():
        if isinstance(value, dict) and value.get("__ndarray__"):
            result[key] = np.array(value["data"], dtype=value["dtype"])
        else:
            result[key] = value
    return result
