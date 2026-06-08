import os
import secrets
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    Response,
    Form,
    UploadFile,
    File,
    Depends,
    status,
    APIRouter,
)
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ValidationError

from paimon_web import cfg
from paimon_web.observability.fs_reader import fs_reader
from paimon_web.observability.run_index import run_index
from paimon_web.observability.process_manager import save_uploaded_files
from paimon_web.observability.process_manager import process_manager as pm
from paimon_web.observability.chat_cache import chat_cache

from paimon_web.observability.reconcile_loop import reconcile_loop
from paimon_web.observability.models import (
    RunSummary,
    RunDetail,
    SubtaskDetail,
    ProcessStatus,
    LaunchRequest,
)
import paimon_web.observability.view_adapter as view_adapter
from paimon_web.api.error_handler import handle_errors
from paimon_web.util.log import debug, info, warning


security = HTTPBasic()

USERNAME = os.environ["PAIMON_BASIC_USER"]
PASSWORD = os.environ["PAIMON_BASIC_PASS"]


def basic_auth(credentials: HTTPBasicCredentials = Depends(security)):
    if not (
        secrets.compare_digest(credentials.username, USERNAME)
        and secrets.compare_digest(credentials.password, PASSWORD)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            headers={"WWW-Authenticate": "Basic"},
        )


router = APIRouter(dependencies=[Depends(basic_auth)])


reconcile_stop_event: asyncio.Event | None = None
reconcile_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global reconcile_stop_event, reconcile_task

    info("[api] Starting worker pool")
    await pm.start_pool()

    info("[api] Starting reconcile loop")
    reconcile_stop_event = asyncio.Event()
    reconcile_task = asyncio.create_task(reconcile_loop(reconcile_stop_event))

    yield

    info("[api] Stopping reconcile loop")
    if reconcile_stop_event:
        reconcile_stop_event.set()
    if reconcile_task:
        reconcile_task.cancel()
        try:
            await reconcile_task
        except asyncio.CancelledError:
            pass

    info("[api] Cleaning up process manager")
    await pm.cleanup()


app = FastAPI(title="Paimon Web Interface", debug=cfg.debug, lifespan=lifespan)
templates_dir = Path(__file__).parent.parent / "web" / "templates"
static_dir = Path(__file__).parent.parent / "web" / "static"
templates = Jinja2Templates(directory=str(templates_dir))

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


def _get_process_status(env_id: str) -> ProcessStatus | None:
    """Get process status for a run if it has a live process."""
    proc_info = pm.get_status(env_id)
    if not proc_info:
        return None
    return ProcessStatus(
        is_live=proc_info.returncode is None,
        pid=proc_info.pid,
        started_at=proc_info.started_at,
        returncode=proc_info.returncode,
    )


# API Endpoints


@router.get("/api/runs", response_model=list[RunSummary])
def list_runs():
    """List all runs from DB index."""
    debug("[api] GET /api/runs")
    runs = []
    for row in run_index.list_runs():
        try:
            summary = RunSummary.model_validate(row)
            summary.process_status = _get_process_status(summary.env_id)
            runs.append(summary)
        except ValidationError as e:
            warning(f"[api] Invalid run row skipped: {e}")
    return runs


@router.get("/api/runs/{env_id}", response_model=RunDetail)
async def get_run(env_id: str):
    """Get run details from filesystem, status from DB."""
    debug(f"[api] GET /api/runs/{env_id}")

    row = run_index.get_run(env_id)
    if not row:
        raise HTTPException(404, f"Run {env_id} not found")

    try:
        run = await fs_reader.get_run_detail(env_id)
    except (FileNotFoundError, ValueError) as e:
        debug(f"[api] FS read failed for {env_id}: {e}")
        raise HTTPException(404, f"Run detail not found: {env_id}")

    run.process_status = _get_process_status(env_id)
    return run


def _coerce_value(val: str) -> str | int | float | bool:
    """Try to parse string as bool/int/float, fallback to str."""
    if val.lower() in ("true", "false"):
        return val.lower() == "true"
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


@router.post("/api/launch")
async def launch_run(
    request: Request,
    response: Response,
    task_name: str = Form(...),
    agent: str | None = Form(None),
    llm: str | None = Form(None),
    reasoning: str | None = Form(None),
    user_preference: str | None = Form(None),
):
    """Launch new run as subprocess with optional file uploads."""
    info(f"[api] POST /api/runs: {task_name[:50]}...")

    launch_req = LaunchRequest(
        task_name=task_name,
        agent=agent,
        llm=llm,
        reasoning=reasoning,
        user_preference=user_preference or None,
    )
    config = launch_req.model_dump(exclude_none=True)

    # Parse extra key-value pairs from form
    form_data = await request.form()
    extra: dict[str, str | int | float | bool] = {}
    idx = 0
    while f"extra_key_{idx}" in form_data:
        key = str(form_data[f"extra_key_{idx}"]).strip()
        val = str(form_data[f"extra_val_{idx}"]).strip()
        if key:
            extra[key] = _coerce_value(val)
        idx += 1
    if extra:
        config["extra"] = extra

    if agent == "Orchestrator":
        config["workflow_type"] = "orchestrator"

    env_id = await pm.launch(config)
    info(f"[api] Launched run {env_id}")

    response.headers["HX-Redirect"] = f"/runs/{env_id}/workspace"
    return {"env_id": env_id, "status": "launched"}


@router.get("/api/runs/{env_id}/process", response_model=ProcessStatus)
def get_process_status(env_id: str):
    """Get live process status."""
    debug(f"[api] GET /api/runs/{env_id}/process")
    status = _get_process_status(env_id)
    return status or ProcessStatus(is_live=False)


@router.post("/api/runs/{env_id}/terminate")
async def terminate_run(env_id: str):
    """Terminate running process."""
    info(f"[api] POST /api/runs/{env_id}/terminate")
    success = await pm.terminate(env_id)
    if not success:
        raise HTTPException(404, "Process not found or already terminated")
    return {"status": "terminated"}


class ChatMessage(BaseModel):
    message: str
    files: list[str] | None = None  # File paths from upload endpoint


@router.post("/api/runs/{env_id}/files")
async def upload_files(env_id: str, files: list[UploadFile] = File(...)):
    """Upload files for mid-chat use. Returns saved file paths."""
    info(f"[api] POST /api/runs/{env_id}/files: {len(files)} files")

    saved_paths = save_uploaded_files(env_id, files)
    if not saved_paths:
        raise HTTPException(400, "No valid files uploaded")

    return {"files": saved_paths}


@router.post("/api/runs/{env_id}/chat/stream")
async def stream_chat(env_id: str, chat_msg: ChatMessage):
    """Send message and stream response via SSE."""
    import json

    info(f"[api] POST /api/runs/{env_id}/chat/stream: {chat_msg.message[:50]}...")

    async def event_generator():
        accumulated_events = []
        try:
            async for chunk in pm.send_message_stream(
                env_id, chat_msg.message, files=chat_msg.files
            ):
                yield f"data: {chunk}\n\n"

                # Accumulate events for cache update
                try:
                    event = json.loads(chunk)
                    if event.get("type") == "event":
                        accumulated_events.append(event)
                except (json.JSONDecodeError, KeyError):
                    pass

            # Update cache after streaming completes
            if accumulated_events:
                try:
                    all_events = fs_reader.get_events(env_id)
                    await chat_cache.update(env_id, all_events)
                    debug(f"[api] Updated cache for {env_id} after stream")
                except Exception as e:
                    debug(f"[api] Failed to update cache after stream: {e}")

        except Exception as e:
            warning(f"[api] Error in stream_chat: {e}")
            yield f"data: error: {str(e)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/api/runs/{env_id}/subtasks/{dir_name}", response_model=SubtaskDetail
)
def get_subtask(env_id: str, dir_name: str):
    """Get subtask details."""
    debug(f"[api] GET /api/runs/{env_id}/subtasks/{dir_name}")
    try:
        return fs_reader.get_subtask(env_id, dir_name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@router.get(
    "/api/runs/{env_id}/subtasks/{dir_name}/files/{filepath:path}/structure"
)
def get_file_as_structure(env_id: str, dir_name: str, filepath: str):
    """Get structure file for 3Dmol visualization."""
    debug(f"[api] GET structure {filepath} from {dir_name}")
    try:
        result = fs_reader.read_structure(env_id, dir_name, filepath)
        return {"filename": filepath, **result}
    except Exception as e:
        raise HTTPException(400, f"Failed to read structure: {e}")


@router.get(
    "/api/runs/{env_id}/subtasks/{dir_name}/files/{filepath:path}/download"
)
def download_file(env_id: str, dir_name: str, filepath: str):
    """Download output file."""
    debug(f"[api] Download file {filepath} from {dir_name}")
    try:
        file_path = fs_reader.get_output_file_path(env_id, dir_name, filepath)
        # Extract just the filename for download header
        filename = Path(filepath).name
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="application/octet-stream",
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@router.get("/api/runs/{env_id}/subtasks/{dir_name}/files/{filepath:path}")
def get_file_content(env_id: str, dir_name: str, filepath: str):
    """Get output file content."""
    debug(f"[api] GET file {filepath} from {dir_name}")
    try:
        content, is_text = fs_reader.read_output_file(env_id, dir_name, filepath)
        return {"filename": filepath, "content": content, "is_text": is_text}
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@router.get("/api/runs/{env_id}/chat")
def get_planner_chat(env_id: str):
    """Get planner chat history."""
    debug(f"[api] GET /api/runs/{env_id}/chat")
    try:
        messages = view_adapter.chat(fs_reader.get_planner_chat(env_id))
        return messages
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(404, str(e))


@router.get("/api/runs/{env_id}/subtasks/{dir_name}/chat")
def get_agent_chat(env_id: str, dir_name: str):
    """Get agent chat history for subtask."""
    debug(f"[api] GET /api/runs/{env_id}/subtasks/{dir_name}/chat")
    try:
        messages = view_adapter.chat(fs_reader.get_agent_chat(env_id, dir_name))
        return messages
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@router.get("/api/runs/{env_id}/events")
async def get_chat_events(env_id: str, offset: int = 0):
    """Get chat events for polling. Returns events from offset."""
    try:
        fs_events = fs_reader.get_events(env_id)
        all_events, _ = await chat_cache.get_or_update(env_id, fs_events)
        display = view_adapter.strip_user_msg_tags(all_events)
        handle = pm.get_status(env_id)
        alive = handle is not None and handle.returncode is None
        return {
            "total": len(display),
            "events": display[offset:],
            "process_alive": alive,
        }
    except (FileNotFoundError, ValueError):
        return {"total": 0, "events": [], "process_alive": False}


# HTML Routes


@router.get("/launch", response_class=HTMLResponse)
@handle_errors
def launch_page(request: Request):
    """Launch page."""
    debug("[api] GET /launch (HTML)")
    return templates.TemplateResponse(
        "unified_run.html", {"request": request, "env_id": None}
    )


@router.get("/runs/{env_id}/workspace", response_class=HTMLResponse)
@handle_errors
async def workspace_page(request: Request, env_id: str):
    """Workspace page for existing run."""
    debug(f"[api] GET /runs/{env_id}/workspace (HTML)")

    row = run_index.get_run(env_id)
    if not row:  # There is no run in DB
        raise HTTPException(404, "Run not found")

    chat_events = []
    try:
        fs_events = fs_reader.get_events(env_id)
        chat_events, used_cache = await chat_cache.get_or_update(env_id, fs_events)
        if used_cache:
            debug(f"[api] Using cached events for {env_id}")
    except (FileNotFoundError, ValueError):
        pass

    # Strip system-appended tags from user messages for display
    display_events = view_adapter.strip_user_msg_tags(chat_events)

    return templates.TemplateResponse(
        "unified_run.html",
        {"request": request, "env_id": env_id, "chat_events": display_events},
    )


@router.get("/runs/{env_id}/error", response_class=HTMLResponse)
@handle_errors
def error_page(request: Request, env_id: str):
    """Show early error details for a run that died before ready."""
    debug(f"[api] GET /runs/{env_id}/error (HTML)")
    early_err = pm.get_effective_status(env_id).early_error
    row = run_index.get_run(env_id)
    task_name = row.get("task_name", "") if row else ""
    return templates.TemplateResponse(
        "error.html",
        {
            "request": request,
            "error": (f"Run '{task_name}' ({env_id}) failed during initialization."),
            "traceback": early_err or "No logs available.",
        },
    )


@router.get(
    "/runs/{env_id}/detail-status-fragment",
    response_class=HTMLResponse,
)
@handle_errors
async def detail_status_fragment(request: Request, response: Response, env_id: str):
    """Status section fragment for polling. Returns file count in header."""
    debug(f"[api] GET /runs/{env_id}/detail-status-fragment")

    es = pm.get_effective_status(env_id)

    if es.early_error:
        return HTMLResponse(
            "",
            headers={"HX-Redirect": f"/runs/{env_id}/error"},
        )

    if not es.ready:
        row = run_index.get_run(env_id)
        task_name = row.get("task_name", "") if row else ""
        return templates.TemplateResponse(
            "detail_status_fragment.html",
            {
                "request": request,
                "env_id": env_id,
                "run": None,
                "run_config": {},
                "status": es.status,
                "task": task_name,
                "process_status": _get_process_status(env_id),
                "token_usage": None,
                "is_waiting": es.status == "pending",
            },
        )

    try:
        run = await fs_reader.get_run_detail(env_id)
    except ValueError:
        raise HTTPException(404, "Run not found")

    run.process_status = _get_process_status(env_id)
    run.status = es.status

    file_count = 0
    try:
        if (
            len(run.subtasks) == 1
            and run.subtasks[0].name.lower() == "working directory"
        ):
            subtask = fs_reader.get_subtask(env_id, run.subtasks[0].dir_name)
            file_count = len(subtask.output_files)
    except (FileNotFoundError, ValueError):
        file_count = 0
    response.headers["X-File-Count"] = str(file_count)

    return templates.TemplateResponse(
        "detail_status_fragment.html",
        {
            "request": request,
            "env_id": env_id,
            "run": run,
            "run_config": run.config.model_dump(),
            "status": run.status,
            "task": run.task_name,
            "process_status": run.process_status,
            "token_usage": run.token_usage,
            "is_waiting": run.status == "pending",
        },
        headers={
            "X-File-Count": str(file_count),
        },
    )


@router.get("/runs/{env_id}/detail-files-fragment", response_class=HTMLResponse)
@handle_errors
async def detail_files_fragment(
    request: Request, env_id: str, subtask: str | None = None
):
    """Files section fragment. Loaded once, refreshed on demand."""
    debug(f"[api] GET /runs/{env_id}/detail-files-fragment")

    row = run_index.get_run(env_id)
    if not row:
        raise HTTPException(404, "Run not found")

    es = pm.get_effective_status(env_id)
    if not es.ready:
        return HTMLResponse("<p class='no-files'>Waiting for run to start...</p>")

    run = await fs_reader.get_run_detail(env_id)

    single_subtask = None
    selected_dir_name = None

    if subtask:
        single_subtask = fs_reader.get_subtask(env_id, subtask)
        selected_dir_name = subtask
    elif (
        len(run.subtasks) == 1
        and run.subtasks[0].name.lower() == "working directory"
    ):
        try:
            single_subtask = fs_reader.get_subtask(
                env_id, run.subtasks[0].dir_name
            )
        except FileNotFoundError:
            pass

    return templates.TemplateResponse(
        "detail_files_fragment.html",
        {
            "request": request,
            "env_id": env_id,
            "run": run,
            "single_subtask": single_subtask,
            "selected_dir_name": selected_dir_name,
        },
    )


@router.get("/runs/{env_id}/chat", response_class=HTMLResponse)
@handle_errors
def planner_chat_view(request: Request, env_id: str):
    """Planner chat history page."""
    debug(f"[api] GET /runs/{env_id}/chat (HTML)")
    raw_messages = fs_reader.get_planner_chat(env_id)
    trajectory = view_adapter.extract_trajectory(raw_messages)
    messages = view_adapter.chat(raw_messages)
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "messages": messages,
            "trajectory": trajectory,
            "env_id": env_id,
            "title": "Main memory",
        },
    )


@router.get("/runs/{env_id}/subtasks/{dir_name}", response_class=HTMLResponse)
@handle_errors
def subtask_detail(request: Request, env_id: str, dir_name: str):
    """Subtask detail page."""
    debug(f"[api] GET /runs/{env_id}/subtasks/{dir_name} (HTML)")
    subtask = fs_reader.get_subtask(env_id, dir_name)
    return templates.TemplateResponse(
        "subtask.html",
        {"request": request, "subtask": subtask, "env_id": env_id},
    )


@router.get(
    "/runs/{env_id}/subtasks/{dir_name}/chat", response_class=HTMLResponse
)
@handle_errors
def agent_chat_view(
    request: Request, env_id: str, dir_name: str
):
    """Agent chat history page."""
    debug(
        f"[api] GET /runs/{env_id}/subtasks/{dir_name}/chat"
        " (HTML)"
    )
    info = fs_reader.find_subtask_info(env_id, dir_name)
    raw_messages = fs_reader.get_agent_chat(env_id, dir_name)
    trajectory = view_adapter.extract_trajectory(raw_messages)
    messages = view_adapter.chat(raw_messages)
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "messages": messages,
            "trajectory": trajectory,
            "env_id": env_id,
            "dir_name": dir_name,
            "subtask_name": info.name,
            "title": f"Agent memory: {info.name}",
        },
    )


@router.get("/", response_class=HTMLResponse)
@handle_errors
def index(
    request: Request,
    status: str | None = None,
    search: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
):
    """Run list page with optional filters."""
    debug(f"[api] GET / status={status} search={search} from={date_from}")

    rows = run_index.list_runs_filtered(
        status=status,
        search=search,
        date_from=date_from,
        date_to=date_to,
    )
    runs = []
    for row in rows:
        try:
            summary = RunSummary.model_validate(row)
            summary.process_status = _get_process_status(summary.env_id)
            runs.append(summary)
        except ValidationError as e:
            warning(f"[api] Invalid run row skipped: {e}")

    statuses = run_index.get_distinct_statuses()

    return templates.TemplateResponse(
        "runs.html",
        {
            "request": request,
            "runs": runs,
            "statuses": statuses,
            "current_status": status,
            "current_search": search,
            "current_date_from": date_from,
            "current_date_to": date_to,
        },
    )


app.include_router(router)
