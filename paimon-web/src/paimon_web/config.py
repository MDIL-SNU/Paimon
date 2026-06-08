import os
from pathlib import Path
from dataclasses import dataclass

import yaml


@dataclass
class WebConfig:
    """Configuration for web interface"""

    debug: bool = True
    log_dir: str = "/tmp"

    paimon_user: str = "paimon"
    paimon_host: str = "localhost"
    paimon_port: int = 22
    paimon_wd: str = ""  # HPC server working dir (/home/paimion/wd)

    web_host: str = "0.0.0.0"
    web_port: int = 8080
    index_db_path: str = ""  # Mandatory. path to SQLite index database


def load_config_from_yaml(path: str | None = None) -> WebConfig:
    """
    Initialize Paimon config from yaml.

    Parameters
    ----------
    path
        path to yaml. if not given, use default config settings

    Returns
    -------
    PaimonConfig
    """
    if path is None:
        yaml_path = str(
            os.getenv("PAIMON_YAML", os.path.expanduser("~/.config/paimon.yaml"))
        )
        path = yaml_path
    with open(path, "r") as f:
        config = yaml.safe_load(f).get("web")

    cfg = WebConfig(**config)

    # validation
    wd = Path(cfg.paimon_wd) if cfg.paimon_wd else None
    if (
        wd is None
        or not wd.exists()
        or not wd.is_dir()
        or not os.access(wd, os.R_OK)
    ):
        raise PermissionError(
            "Paimon wd must be an existing directory with read permission"
        )

    log_dir = Path(cfg.log_dir) if cfg.log_dir else None
    if (
        log_dir is None
        or not log_dir.exists()
        or not log_dir.is_dir()
        or not os.access(log_dir, os.W_OK)
    ):
        raise PermissionError(
            "Log dir must be an existing directory with write permission"
        )

    return cfg
