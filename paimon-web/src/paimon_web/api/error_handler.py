import inspect
import traceback
from functools import wraps

from fastapi import Request
from fastapi.templating import Jinja2Templates
from pathlib import Path

from paimon_web.util.log import debug, warning


templates_dir = Path(__file__).parent.parent / "web" / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


def handle_errors(func):
    """Decorator to handle errors and show full traceback in error template"""

    is_async = inspect.iscoroutinefunction(func)

    @wraps(func)
    async def wrapper(request: Request, *args, **kwargs):
        try:
            if is_async:
                return await func(request, *args, **kwargs)
            else:
                return func(request, *args, **kwargs)

        except FileNotFoundError as e:
            error_msg = str(e)
            tb = traceback.format_exc()
            warning(f"[error_handler] FileNotFoundError in {func.__name__}: {error_msg}")
            debug(f"[error_handler] Traceback: {tb}")
            return templates.TemplateResponse(
                "error.html",
                {"request": request, "error": error_msg, "traceback": tb},
                status_code=404,
            )

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            tb = traceback.format_exc()
            warning(f"[error_handler] Exception in {func.__name__}: {error_msg}")
            debug(f"[error_handler] Traceback: {tb}")
            return templates.TemplateResponse(
                "error.html",
                {"request": request, "error": error_msg, "traceback": tb},
                status_code=500,
            )

    return wrapper
