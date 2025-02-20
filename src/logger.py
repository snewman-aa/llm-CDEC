from loguru import logger
import sys

def setup_logger(log_level="INFO"):
    """
    Sets up and configures the Loguru logger.

    Args:
        log_level (str): The logging level to use (e.g., "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
                         Defaults to "INFO".
    """
    logger.remove()  # Remove default handler to avoid duplication

    # Add a new handler to output to console (sys.stdout)
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{module: <8}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )
    return logger # While loguru uses a global logger, returning it for clarity
