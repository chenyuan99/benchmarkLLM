from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="benchmarkLLM",
    version="0.1.0",
    author="Yuan Chen",
    author_email="Yuan.Chen@sojoai.com",
    description="A benchmarking tool for LM Studio inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chenyuan99/benchmarkLLM",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.31.0",
        "psutil>=5.9.5",
        "numpy>=1.26.0",
        "pandas>=2.1.0",
        "matplotlib>=3.8.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.65.0"
    ],
    entry_points={
        "console_scripts": [
            "lmstudio-benchmark=lmstudio_benchmark.main:main",
        ],
    },
)
