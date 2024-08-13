"""
   Written by ChaptGPT in response to the prompt: Can you suggest an easy way to
   colorize logging output in Python? Aug 10,2024
"""

import logging
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Custom formatter class
class ColoredFormatter(logging.Formatter):
    COLORS = {
        logging.INFO: Fore.RED,
        logging.DEBUG: Fore.GREEN,
        logging.WARNING: Fore.CYAN,
        logging.ERROR: Fore.YELLOW,
        logging.CRITICAL: Fore.MAGENTA,
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelno, Fore.WHITE)
        log_message = super().format(record)
        return f"{log_color}{log_message}{Style.RESET_ALL}"

# Example usage
def get_logger(name):
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    formatter = ColoredFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger
