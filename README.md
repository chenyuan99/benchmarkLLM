# LM Studio Inference Benchmark

This project provides tools to benchmark inference performance for LM Studio models. It measures various metrics including:
- Inference latency
- Tokens per second
- Memory usage
- Response generation time

## Requirements
- Python 3.8+
- LM Studio running locally
- Required Python packages (see requirements.txt)

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python main.py --model "your_model_name" --prompt "your_test_prompt"
```

## Features
- Measures inference latency across different prompt lengths
- Supports multiple model comparisons
- Generates detailed performance reports
- Memory usage tracking
- Configurable test parameters

## License
MIT License
