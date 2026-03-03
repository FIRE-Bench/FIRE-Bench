#!/usr/bin/env python3
"""
Quick start script for LLM Evaluation Pipeline
"""

import os
import sys
from pathlib import Path

# 禁用 __pycache__ 生成
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.cli import CLIRunner


def quick_start():
    """Quick start guide"""
    print("=== LLM Evaluation Pipeline - Quick Start ===\n")
    
    print("1. List available datasets:")
    print("   python run_evaluation.py --list-datasets\n")
    
    print("2. Run evaluation (example):")
    print("   python run_evaluation.py \\")
    print("     --url https://api.openai.com/v1 \\")
    print("     --api-key sk-your-api-key \\")
    print("     --model gpt-4 \\")
    print("     --model-name gpt-4 \\")
    print("     --datasets FIRE \\")
    print("     --max-samples 5\n")
    
    print("3. Get help:")
    print("   python run_evaluation.py --help\n")
    
    print("Notes:")
    print("- Ensure dataset files exist in the dataset/ directory")
    print("- Keep your API keys secure and do not expose them")
    print("- Test with --max-samples parameter first")


def main():
    """Main entry point"""
    if len(sys.argv) == 1:
        # No arguments provided, show quick start guide
        quick_start()
        return
    
    # Run CLI
    runner = CLIRunner()
    runner.run()


if __name__ == "__main__":
    main() 