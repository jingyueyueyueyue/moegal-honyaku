import logging
import sys
from colorlog import ColoredFormatter

from app.core.paths import LOGS_DIR

LOGS_DIR.mkdir(parents=True, exist_ok=True)

LOG_LEVEL_NAME = (sys.modules.get("os") or __import__("os")).getenv("MOEGAL_LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL_NAME, logging.INFO)

logger = logging.getLogger("moegal")
logger.setLevel(LOG_LEVEL)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(LOG_LEVEL)

console_formatter = ColoredFormatter(
    "%(log_color)s%(levelname)s%(reset)s:     %(message)s",
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
)
console_handler.setFormatter(console_formatter)


file_handler = logging.FileHandler(LOGS_DIR / "app.log", encoding='utf-8')
file_handler.setLevel(LOG_LEVEL)

file_formatter = logging.Formatter(
    "%(name)s, %(levelname)s, %(asctime)s, %(filename)s:%(lineno)d, %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(file_formatter)

logger.handlers.clear()
logger.addHandler(console_handler)
logger.addHandler(file_handler)
