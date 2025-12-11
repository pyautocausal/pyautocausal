import logging
import sys

def get_class_logger(cls_name: str) -> logging.Logger:
    """
    Creates a logger instance for a specific class.
    
    Args:
        cls_name: Name of the class requesting the logger
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(cls_name)
    
    if not logger.handlers:
        # Create handlers
        console_handler = logging.StreamHandler(sys.stdout)
        
        # Create formatters and add it to handlers
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(log_format)
        console_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        logger.addHandler(console_handler)
        logger.setLevel(logging.DEBUG)  # Set to lowest level to capture all messages
    
    return logger
