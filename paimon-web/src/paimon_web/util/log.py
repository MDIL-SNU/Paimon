import logging
from datetime import datetime

logger = logging.getLogger(__name__)
_handler = logging.StreamHandler()
logger.addHandler(_handler)

logger.setLevel(level=logging.DEBUG)


def debug(any) -> None:
    now = datetime.now().isoformat(timespec='seconds')
    logger.debug(f"[{now}][DEBUG] {any}")


def info(any) -> None:
    now = datetime.now().isoformat(timespec='seconds')
    logger.debug(f"[{now}][INFO] {any}")


def warning(any) -> None:
    now = datetime.now().isoformat(timespec='seconds')
    logger.debug(f"[{now}][WARNING] {any}")
