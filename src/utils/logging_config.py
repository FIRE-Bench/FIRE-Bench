"""
Logging configuration for the evaluation pipeline
"""

import logging
import os
import sys
from loguru import logger


def setup_logging(verbose: bool = False):
    """Setup logging configuration with HTTP request logging disabled"""
    
    # Suppress HTTP request related logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("azure").setLevel(logging.WARNING)
    
    # Set environment variables
    os.environ["HTTPX_LOG_LEVEL"] = "warning"
    os.environ["URLLIB3_LOG_LEVEL"] = "warning"
    os.environ["OPENAI_LOG_LEVEL"] = "warning"
    
    # Configure loguru
    logger.remove()  # Remove default handler
    
    level = "DEBUG" if verbose else "INFO"
    format_str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    
    logger.add(sys.stdout, format=format_str, level=level, colorize=True)


def disable_http_logging():
    """Disable HTTP request logging globally"""
    import logging
    
    # Suppress HTTP request related logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("azure").setLevel(logging.WARNING)
    
    # Set environment variables
    os.environ["HTTPX_LOG_LEVEL"] = "warning"
    os.environ["URLLIB3_LOG_LEVEL"] = "warning"
    os.environ["OPENAI_LOG_LEVEL"] = "warning" 