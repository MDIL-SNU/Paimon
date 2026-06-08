import logging
import traceback
import json
import pickle
from datetime import datetime

from paimon import cfg

logger = logging.getLogger(__name__)
_handler = logging.StreamHandler()
logger.addHandler(_handler)

if cfg.debug:
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


def debug_var(var, name, save_json: bool = False) -> None:
    logger.debug(f"[DEBUG] {name}: type: {type(var)}, print: {var}, dir: {dir(var)}")

    if save_json:
        with open(f"{name}.json", "w") as f:
            try:
                content = json.dumps(var.model_dump(), indent=2)
            except Exception as e1:
                with open(f"{name}.pkl", "wb") as f2:
                    try:
                        pickle.dump(var, f2)
                    except Exception as e2:
                        logger.debug(f"[DEBUG] {name} model_dump FAILED use str")
                        try:
                            content = str(var)
                        except Exception as e2:
                            logger.debug(f"[DEBUG] {name} str FAILED use repr")
                            content = repr(var)
            f.write(content)
        logger.debug(f"{name}.json is saved")


def debug_assert(cond: bool, msg: str, raise_exc: bool = True) -> None:
    """
    Assert a condition in debug mode, logging a backtrace if it fails.

    Parameters:
    - cond: The condition to assert.
    - msg: Message to log or include in the raised exception.
    - raise_exc: If True, raises AssertionError on failure; otherwise only logs.
    """
    if cond:
        return

    # Capture the current stack (excluding this function call)
    stack = "".join(traceback.format_stack()[:-1])
    logger.error(f"debug_assert failed: {msg}\nBacktrace:\n{stack}")

    if raise_exc:
        raise AssertionError(msg)
