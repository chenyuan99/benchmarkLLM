import argparse
import json
import time
import requests
import psutil
import pandas as pd
from tqdm import tqdm

class LMStudioBenchmark:
    def __init__(self, host="localhost", port=1234):
        self.base_url = f"http://{host}:{port}/v1"
        self.results = []

    def check_connection(self):
        try:
            response = requests.get(f"{self.base_url}/models")
            return response.status_code == 200
        except requests.RequestException:
            return False

    def measure_inference(self, prompt, max_tokens=100):
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": False
        }

        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            result = response.json()
            generated_text = result['choices'][0]['message']['content']
            total_tokens = result.get('usage', {}).get('total_tokens', 0)
            
            return {
                'prompt_length': len(prompt),
                'response_length': len(generated_text),
                'total_tokens': total_tokens,
                'latency': end_time - start_time,
                'tokens_per_second': total_tokens / (end_time - start_time),
                'memory_usage_mb': end_memory - start_memory,
                'success': True
            }
            
        except Exception as e:
            return {
                'prompt_length': len(prompt),
                'error': str(e),
                'success': False
            }

    def run_benchmark(self, prompts, iterations=3):
        if not self.check_connection():
            print("Error: Cannot connect to LM Studio. Please ensure it's running.")
            return

        print(f"Running benchmark with {len(prompts)} prompts, {iterations} iterations each...")
        
        for prompt in prompts:
            for _ in tqdm(range(iterations), desc=f"Testing prompt of length {len(prompt)}"):
                result = self.measure_inference(prompt)
                if result['success']:
                    self.results.append(result)

    def generate_report(self, output_file="benchmark_results.csv"):
        if not self.results:
            print("No results to report!")
            return

        df = pd.DataFrame(self.results)
        df.to_csv(output_file, index=False)
        
        print("\nBenchmark Summary:")
        print(f"Average latency: {df['latency'].mean():.2f} seconds")
        print(f"Average tokens/second: {df['tokens_per_second'].mean():.2f}")
        print(f"Average memory usage: {df['memory_usage_mb'].mean():.2f} MB")
        print(f"\nDetailed results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Benchmark LM Studio inference')
    parser.add_argument('--host', default='localhost', help='LM Studio host')
    parser.add_argument('--port', type=int, default=1234, help='LM Studio port')
    parser.add_argument('--iterations', type=int, default=3, help='Number of iterations per prompt')
    args = parser.parse_args()

    # Sample prompts of varying lengths
    test_prompts = [
        "What is machine learning?",
        "Explain the concept of neural networks and their applications in modern AI systems.",
        "Write a detailed essay about the history of artificial intelligence, its current state, and future prospects. Include specific examples and milestones.",
    ]

    benchmark = LMStudioBenchmark(host=args.host, port=args.port)
    benchmark.run_benchmark(test_prompts, args.iterations)
    benchmark.generate_report()

if __name__ == '__main__':
    main()
