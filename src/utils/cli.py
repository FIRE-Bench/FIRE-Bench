"""
Command Line Interface for the evaluation pipeline
"""

import argparse
import asyncio
from typing import List, Optional
from loguru import logger

from ..core.pipeline import EvaluationPipeline
from .config import ConfigManager
from .logging_config import setup_logging


class CLIRunner:
    """Command line interface runner"""
    
    def __init__(self):
        self.config_manager = None
        self.pipeline = EvaluationPipeline()
        
    def create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser"""
        parser = argparse.ArgumentParser(
            description="LLM Evaluation Pipeline",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""

            """
        )
        
        # Model configuration - core required parameters
        parser.add_argument(
            "--config-file", 
            default="config/datasets.yaml",
            help="Config file for the model and datasets"
        )

        parser.add_argument(
            "--resume-folder", 
            default="",
            help="Resume directory for the evaluation"
        )

        parser.add_argument(
            "--url", 
            nargs="+",
            help="Base URL(s) for the model API (OpenAI compatible). Can specify multiple URLs for load balancing."
        )
        
        parser.add_argument(
            "--api-key",
            default="token1",
            help="API key for the model service"
        )
        
        parser.add_argument(
            "--model",
            default="default",
            help="Model name to openai clinet"
        )

        parser.add_argument(
            "--model-name",
            default="default",
            help="alias model name to use, for storage the result"
        )

        parser.add_argument(
            "--api-type",
            default="default",
            help="API type for the model service"
        )

        parser.add_argument(
            "--api-version",
            default="default",
            help="API version for the model service"
        )

        parser.add_argument(
            "--per-url-max-workers",
            type=int,
            default=128,
            help="Maximum number of workers per URL (default: 128)"
        )
        
        # Dataset selection - mutually exclusive
        dataset_group = parser.add_mutually_exclusive_group()
        dataset_group.add_argument(
            "--datasets",
            nargs="+",
            help="List of JSON dataset names to evaluate on"
        )
        
        # Optional parameters
        parser.add_argument(
            "--max-samples",
            type=int,
            help="Maximum number of samples to evaluate (for testing)"
        )
        
        parser.add_argument(
            "--results-dir",
            default="results",
            help="Directory to save evaluation results (default: results)"
        )
        
        # Utility commands
        parser.add_argument(
            "--list-datasets",
            action="store_true",
            help="List all available datasets"
        )
        
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable verbose logging"
        )
        
        # Request mode configuration
        # Use custom function to properly handle boolean parameters
        def str2bool(v):
            if isinstance(v, bool):
                return v
            if v.lower() in ('yes', 'true', 't', 'y', '1'):
                return True
            elif v.lower() in ('no', 'false', 'f', 'n', '0'):
                return False
            else:
                raise argparse.ArgumentTypeError('Boolean value expected.')
        
        parser.add_argument(
            "--streaming",
            type=str2bool,
            default=False,
            help="Enable streaming response (default: False)"
        )
        
        parser.add_argument(
            "--use-chat",
            type=str2bool,
            default=True,
            help="Use chat completion (default: True)"
        )
        
        return parser
    
    def setup_logging(self, verbose: bool = False):
        """Setup logging configuration"""
        # Use new logging configuration function
        setup_logging(verbose)
    
    def list_datasets(self):
        """List all available datasets"""
        datasets_info = self.config_manager.list_datasets_info()
        
        print("\n=== Available Datasets ===")
        
        # Group by type
        datasets = []
        
        for name, info in datasets_info.items():
            
            datasets.append((name, info))
        
        if datasets:
            print("\n📄 Datasets:")
            for name, info in datasets:
                print(f"  - {name}: {info['description']}")
        
    
    async def run_evaluation(self, args):
        """Run evaluation on datasets"""
        if not args.datasets:
            logger.error("Please specify --datasets for evaluation")
            return
        
        # Validate datasets are dataset type
        invalid_datasets = []
        for dataset_name in args.datasets:
            config = self.config_manager.get_dataset_config(dataset_name)   # type: ignore
            if not config:
                invalid_datasets.append(dataset_name)
        
        if invalid_datasets:
            logger.error(f"Invalid or non datasets: {invalid_datasets}")
            return
        
        # Create model configuration
        
        model_config = self.config_manager.create_model_config(  # type: ignore
            name=args.model_name,
            urls=args.url,  # Pass the list of URLs
            api_key=args.api_key,
            api_type=args.api_type,
            api_version=args.api_version,
            model=args.model,
            per_url_max_workers=args.per_url_max_workers,
            streaming=args.streaming,
            use_chat=args.use_chat
        )
        
        # Run evaluation
        results = await self.pipeline.run_evaluation(
            config_manager=self.config_manager,  # type: ignore
            model_config=model_config,
            dataset_names=args.datasets,
            max_samples=args.max_samples,
            results_dir=args.results_dir,
            resume_folder=args.resume_folder
        )
        
        # Display results
        print("\n=== Evaluation Results ===")
        for result in results:
            print(f"{result.dataset_name}: {result.metrics}")

    
    def run(self, args: Optional[List[str]] = None):
        """Main entry point"""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        if str(parsed_args.resume_folder):
            config_file = f"{parsed_args.resume_folder}/datasets.yaml"
            parsed_args.config_file = config_file
        self.config_manager = ConfigManager(parsed_args.config_file)        
        # Setup logging
        self.setup_logging(parsed_args.verbose)
        
        # Handle utility commands
        if parsed_args.list_datasets:
            self.list_datasets()
            return
        
        if not parsed_args.url or not parsed_args.api_key:
            logger.error("Please provide --url (at least one) and --api-key for evaluation")
            return
        
        # Run evaluation
        try:
            if parsed_args.datasets:
                asyncio.run(self.run_evaluation(parsed_args))
        except KeyboardInterrupt:
            logger.info("Evaluation interrupted by user")
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            if parsed_args.verbose:
                raise