from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="time_series_compression",
    version="2.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "PyWavelets>=1.1.1",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.2",
    ],
    extras_require={
        "plot": ["matplotlib>=3.4.2"],
    },
    author="Moudather Chelbi",
    author_email="moudather.chelbi@gmail.com",
    description="Production-ready time series compression with byte serialization and stateless algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vinerya/time_series_compression",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
)
