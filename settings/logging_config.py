"""Logging configuration for the Vote Efficiency LLM."""
import logging

def setup_logging():
    """Configure logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)

# Create logger instance
logger = setup_logging()