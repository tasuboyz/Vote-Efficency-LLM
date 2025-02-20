import time
import logging
import functools
from typing import Callable, Any, Optional
from beem import Hive, Steem
from ..settings.config import BLOCKCHAIN_CHOICE
from .beem import get_working_node

logger = logging.getLogger(__name__)

def retry_on_blockchain_failure(
    max_retries: int = 3,
    delay: int = 1,
    blockchain_instance: Optional[Any] = None
) -> Callable:
    """
    Decorator that retries blockchain operations on failure with node switching capability.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
        blockchain_instance: Current blockchain instance (Hive or Steem)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal blockchain_instance
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Final attempt failed for {func.__name__}: {str(e)}")
                        raise
                    
                    logger.warning(f"Operation {func.__name__} failed, retrying... ({attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    
                    # Try to get a new node and recreate blockchain instance
                    try:
                        new_node = get_working_node(BLOCKCHAIN_CHOICE)
                        if blockchain_instance:
                            if isinstance(blockchain_instance, Hive):
                                blockchain_instance = Hive(node=new_node)
                            elif isinstance(blockchain_instance, Steem):
                                blockchain_instance = Steem(node=new_node)
                            # Update the blockchain instance in the kwargs if present
                            if 'blockchain_instance' in kwargs:
                                kwargs['blockchain_instance'] = blockchain_instance
                            logger.info(f"Switched to new node: {new_node}")
                    except Exception as node_error:
                        logger.warning(f"Failed to switch node: {str(node_error)}")
            
            return None
        return wrapper
    return decorator

def log_execution_time(func: Callable) -> Callable:
    """
    Decorator that logs the execution time of a function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger.debug(
            f"Function {func.__name__} executed in {end_time - start_time:.2f} seconds"
        )
        return result
    return wrapper

def handle_blockchain_exceptions(func: Callable) -> Callable:
    """
    Decorator that handles common blockchain-related exceptions.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Blockchain operation failed in {func.__name__}: {str(e)}")
            raise
    return wrapper

def validate_blockchain_response(func: Callable) -> Callable:
    """
    Decorator that validates blockchain API responses.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        
        if result is None:
            logger.warning(f"Empty response from {func.__name__}")
            return None
            
        return result
    return wrapper