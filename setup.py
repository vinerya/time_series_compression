from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="time_series_compression",
    version="0.5.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.4.2",
        "scipy>=1.7.0",
        "PyWavelets>=1.1.1",
    ],
    author="Moudather Chelbi",
    author_email="moudather.chelbi@gmail.com",
    description="An extensible framework for time series data compression with multiple algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vinerya/time_series_compression",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
)