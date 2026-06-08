import uvicorn
from pathlib import Path
from paimon_web import cfg
from paimon_web.api.main import app

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    uvicorn.run(
        "paimon_web.api.main:app",
        host=cfg.web_host,
        port=cfg.web_port,
        reload=cfg.debug,
        reload_dirs=[str(project_root)],
    )
