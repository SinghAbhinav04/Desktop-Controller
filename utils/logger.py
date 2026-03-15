import logging
import sys

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a configured logger instance.
    Logs are prefixed with timestamp + module name.
    Supported levels: logging.DEBUG, logging.INFO, logging.WARNING.
    """
    logger = logging.getLogger(name)
    
    # Prevent adding multiple handlers if get_logger is called multiple times for the same module
    if not logger.handlers:
        logger.setLevel(level)
        
        # Create console handler to output to stdout
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        # Format: timestamp - [LEVEL] - module name: message
        formatter = logging.Formatter(
            '%(asctime)s - [%(levelname)s] - %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        
        # Avoid propagating to the root logger to prevent duplicate logs
        logger.propagate = False
        
    return logger
