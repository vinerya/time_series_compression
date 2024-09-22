import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import norm
from scipy.fft import dct, idct
import zlib
import pywt

class CompressionAlgorithm(ABC):
    @abstractmethod
    def compress(self, data):
        pass

    @abstractmethod
    def decompress(self, compressed_data):
        pass

class DifferenceEncoding(CompressionAlgorithm):
    def compress(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a numpy array")
        
        compressed = np.zeros_like(data)
        compressed[0] = data[0]
        compressed[1:] = np.diff(data)
        return compressed

    def decompress(self, compressed_data):
        if not isinstance(compressed_data, np.ndarray):
            raise TypeError("Input data must be a numpy array")
        
        return np.cumsum(compressed_data)

class PAA(CompressionAlgorithm):
    def __init__(self, segments):
        self.segments = segments

    def compress(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a numpy array")
        
        self.original_length = len(data)
        segment_len = len(data) // self.segments
        compressed = np.mean(data[:len(data) - len(data) % segment_len].reshape(-1, segment_len), axis=1)
        return compressed

    def decompress(self, compressed_data):
        if not isinstance(compressed_data, np.ndarray):
            raise TypeError("Input data must be a numpy array")
        
        return np.repeat(compressed_data, self.original_length // self.segments)

class SAX(CompressionAlgorithm):
    def __init__(self, segments, alphabet_size):
        self.segments = segments
        self.alphabet_size = alphabet_size
        self.breakpoints = norm.ppf(np.linspace(0, 1, alphabet_size + 1)[1:-1])

    def compress(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a numpy array")
        
        self.original_length = len(data)
        self.original_mean = np.mean(data)
        self.original_std = np.std(data)
        
        # Normalize the data
        normalized_data = (data - self.original_mean) / self.original_std
        
        # PAA compression
        paa = PAA(self.segments)
        paa_data = paa.compress(normalized_data)
        
        # Discretize to symbols
        symbolic_data = np.digitize(paa_data, self.breakpoints)
        return symbolic_data

    def decompress(self, compressed_data):
        if not isinstance(compressed_data, np.ndarray):
            raise TypeError("Input data must be a numpy array")
        
        # Convert symbols back to PAA
        paa_data = self.breakpoints[compressed_data - 1]
        
        # PAA decompression
        paa = PAA(self.segments)
        paa.original_length = self.original_length
        decompressed = paa.decompress(paa_data)
        
        # Denormalize
        return decompressed * self.original_std + self.original_mean

class DCT(CompressionAlgorithm):
    def __init__(self, keep_coeffs):
        self.keep_coeffs = keep_coeffs

    def compress(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a numpy array")
        
        self.original_shape = data.shape
        dct_coeffs = dct(data)
        compressed = dct_coeffs[:self.keep_coeffs]
        return compressed

    def decompress(self, compressed_data):
        if not isinstance(compressed_data, np.ndarray):
            raise TypeError("Input data must be a numpy array")
        
        full_coeffs = np.zeros(self.original_shape)
        full_coeffs[:len(compressed_data)] = compressed_data
        return idct(full_coeffs, n=self.original_shape[0])

class RunLengthEncoding(CompressionAlgorithm):
    def compress(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a numpy array")
        
        self.original_shape = data.shape
        compressed = []
        count = 1
        for i in range(1, len(data)):
            if data[i] == data[i-1]:
                count += 1
            else:
                compressed.append((data[i-1], count))
                count = 1
        compressed.append((data[-1], count))
        return compressed

    def decompress(self, compressed_data):
        decompressed = []
        for value, count in compressed_data:
            decompressed.extend([value] * count)
        return np.array(decompressed).reshape(self.original_shape)

class ZlibCompression(CompressionAlgorithm):
    def compress(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a numpy array")
        
        self.original_shape = data.shape
        return zlib.compress(data.tobytes())

    def decompress(self, compressed_data):
        decompressed = zlib.decompress(compressed_data)
        return np.frombuffer(decompressed, dtype=np.float64).reshape(self.original_shape)

class DiscreteWaveletTransform(CompressionAlgorithm):
    def __init__(self, wavelet='db4', level=None, threshold=0.1):
        self.wavelet = wavelet
        self.level = level
        self.threshold = threshold

    def compress(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a numpy array")
        
        self.original_shape = data.shape
        coeffs = pywt.wavedec(data, self.wavelet, level=self.level)
        
        # Threshold the coefficients
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], self.threshold * np.max(np.abs(coeffs[i])))
        
        return coeffs

    def decompress(self, compressed_data):
        if not isinstance(compressed_data, list):
            raise TypeError("Input data must be a list of wavelet coefficients")
        
        return pywt.waverec(compressed_data, self.wavelet)[:self.original_shape[0]]

class TimeSeriesCompressor:
    def __init__(self, algorithm=None):
        self.algorithm = algorithm or DifferenceEncoding()

    def set_algorithm(self, algorithm):
        if not isinstance(algorithm, CompressionAlgorithm):
            raise TypeError("Algorithm must be an instance of CompressionAlgorithm")
        self.algorithm = algorithm

    def compress(self, data):
        return self.algorithm.compress(data)

    def decompress(self, compressed_data):
        return self.algorithm.decompress(compressed_data)