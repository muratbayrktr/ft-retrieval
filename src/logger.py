import logging
import sys
from pathlib import Path
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme
from rich.traceback import install as install_rich_traceback

# Install rich traceback handler
install_rich_traceback(show_locals=True)

# Create a custom theme for rich console
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red",
    "debug": "green",
    "critical": "red bold",
    "file": "blue",
    "line": "blue",
})

# Initialize rich console with custom theme
console = Console(theme=custom_theme)

class FileTrackingFormatter(logging.Formatter):
    """Custom formatter that includes file name and line number in the log message."""
    
    def format(self, record):
        # Get the file name and line number
        if hasattr(record, 'pathname'):
            file_path = Path(record.pathname)
            record.file_info = f"[file]{file_path.name}[/file]:[line]{record.lineno}[/line]"
        else:
            record.file_info = ""
            
        return super().format(record)

def setup_logger(name: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with rich formatting and file tracking.
    
    Args:
        name (str, optional): Name of the logger. If None, returns the root logger.
        level (int, optional): Logging level. Defaults to logging.INFO.
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicate logs
    logger.handlers = []
    
    # Create rich handler
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=True,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        markup=True
    )
    
    # Create formatter
    formatter = FileTrackingFormatter(
        fmt="%(asctime)s - %(levelname)s - %(file_info)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Set formatter for rich handler
    rich_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(rich_handler)
    
    return logger

# Create default logger instance
logger = setup_logger("ft_retrieval")

def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance. If name is provided, returns a logger with that name.
    Otherwise returns the default logger.
    
    Args:
        name (str, optional): Name of the logger. If None, returns the default logger.
        
    Returns:
        logging.Logger: Logger instance
    """
    if name is None:
        return logger
    return setup_logger(name)

# Example usage:
if __name__ == "__main__":
    # Test the logger
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Test with a custom logger
    custom_logger = get_logger("custom")
    custom_logger.info("This is a message from a custom logger") 