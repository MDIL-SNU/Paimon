import os
from typing import Literal
import requests
from dataclasses import dataclass, field

import yaml


@dataclass
class CriticConfig:
    use: bool = False
    num_critics: int = 4
    max_turns: int = 2
    need_actions_concern_ratio: float = 0.5
    critic_llm: str = "fast_reasoning"
    # TODO:
    # spokesman_llm: fast_reasoning & low, need more flexible way to manage llms

    def validate(self) -> None:
        if self.max_turns < 1:
            raise ValueError(
                f"CriticConfig.max_turns must be >= 1, got {self.max_turns}"
            )


@dataclass
class RAGConfig:
    """Configuration for RAG systems (LAMMPS and Code Search).

    All RAG systems use a single shared ChromaDB database with separate
    collections for each system (e.g., 'lammps_docs', 'code_ase').
    """

    # Shared ChromaDB settings
    chroma_db_path: str = field(
        default_factory=lambda: os.getenv("CHROMA_DB_PATH", "./chroma_db")
    )

    # LAMMPS RAG settings
    lammps_docs_dir: str = "/tmp/lammps/doc/src"
    embed_model: str = "text-embedding-3-small"
    # For "select_and_quote" and "decompose_task" (which are not used now)
    llm_model: str = "fast_reasoning"
    # vector database hybrid retrieval
    use_hybrid: bool = True

    # annotator settings
    annotator: bool = True
    annotator_LLM: str = "base_reasoning"

    # Code Search RAG settings
    code_search_generate_summary: bool = False
    code_search_crawler_timeout: int = 100


@dataclass(frozen=True)
class LlampConfig:
    """Configuration for llamp service integration"""

    use: bool = False
    server_working_dir: str = ""
    service_url: str = "http://localhost:8000"
    openai_api_key: str | None = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    mp_api_key: str | None = field(default_factory=lambda: os.getenv("MP_API_KEY"))
    timeout: int = 900

    def validate(self, config_path: str | None = None) -> None:
        """Validate configuration when enabled."""
        if not self.use:
            return

        # Check API keys
        missing = []
        if not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        if not self.mp_api_key:
            missing.append("MP_API_KEY")

        if missing:
            keys = ", ".join(missing)
            location = f" in {config_path}" if config_path else ""
            raise ValueError(
                f"LlampConfig{location} is enabled but missing API keys: {keys}. "
                f"Set them as environment variables or in the YAML file."
            )

        # Check service availability
        try:
            response = requests.get(f"{self.service_url}/api/health", timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Cannot connect to LLaMPP service at {self.service_url}. "
                f"Make sure the server is running. Error: {type(e).__name__}"
            ) from e

        swd = self.server_working_dir
        if not swd or not os.path.exists(swd):
            raise ValueError(
                f"Server working directory: {swd} does not exist or not set"
            )
        elif not os.access(swd, os.R_OK):
            raise ValueError(f"Server working directory: {swd} no read permission")
        elif not os.path.isdir(swd):
            raise ValueError("Server working directory is not a directory")


@dataclass(frozen=True)
class WebSearchConfig:
    """Configuration for Tavily web search tool."""

    tavily_api_key: str | None = field(
        default_factory=lambda: os.getenv("TAVILY_API_KEY")
    )

    # config of how Paimon uses Tavily
    executor: bool = False
    planner: bool = False
    interactive: bool = False
    max_results: int = 3

    # config of tavily 'search' end point
    search_depth: Literal["basic", "advanced", "fast", "ultra-fast"] = "basic"

    def validate(self) -> None:
        """Validate configuration when any role enables search."""
        if (self.executor or self.planner) and not self.tavily_api_key:
            raise ValueError("web search enabled but TAVILY_API_KEY is missing.")


@dataclass(frozen=True)
class PaimonConfig:
    """Default Paimon config"""

    debug: bool = False
    debug_preamble: bool = False
    use_arize_phoenix: bool = False
    open_ai_store: bool = True
    open_ai_response_api: bool = True  # TODO: not used
    use_episodic: bool = False

    # Paimon server settings, should lead to GPU debug node
    paimon_user: str = "paimon"
    paimon_host: str = "localhost"
    paimon_port: int = 22

    # Paimon server "login node" settings, which has full access to slurm service
    use_slurm: bool = False
    paimon_slurm_user: str = "paimon"
    paimon_slurm_host: str = "0.0.0.0"
    paimon_slurm_port: int = 22
    slurm_policy: str = "odin"  # TODO: change default
    slurm_parse_cmd: Literal["sacct", "squeue"] = "sacct"
    auto_cancel_pending: bool = False

    # LLM settings. "{api}/{model}"
    fast_llm: str = "openai/gpt-5-mini"
    base_llm: str = "openai/gpt-4.1"

    fast_kwargs: dict = field(default_factory=dict)
    base_kwargs: dict = field(default_factory=dict)

    fast_reasoning_llm: str = "openai/gpt-5-mini"  # executor
    base_reasoning_llm: str = "openai/gpt-5"  # planner

    fast_reasoning_kwargs: dict = field(default_factory=dict)
    base_reasoning_kwargs: dict = field(default_factory=dict)

    # TODO: Should be override-able when init workflows [plan.py]
    critic_config: CriticConfig = field(default_factory=CriticConfig)
    rag_config: RAGConfig = field(default_factory=RAGConfig)
    llamp_config: LlampConfig = field(default_factory=LlampConfig)
    web_search_config: "WebSearchConfig" = field(default_factory=WebSearchConfig)

    # LLM settings
    temperature: float = 1.0
    timeout: float = 1800
    max_retries: int = 2
    cost_file: str | None = None  # 1 USD per 1M tokens

    default_expert_llm: str = "fast_reasoning"

    # Environment
    paimon_envs_root: str = ".local/Paimon/paimon-envs"

    # Interface
    max_agent_iterations = 40
    cli_max_width: int = 79


def load_config_from_yaml(path: str | None = None) -> PaimonConfig:
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
        # Read only yaml under paimon
        config = yaml.safe_load(f).get("paimon")

    critic_config = CriticConfig(**config.pop("critic_config", {}))
    rag_config = RAGConfig(**config.pop("rag_config", {}))
    llamp_config = LlampConfig(**config.pop("llamp_config", {}))
    web_search_config = WebSearchConfig(**config.pop("web_search_config", {}))

    critic_config.validate()
    llamp_config.validate()
    web_search_config.validate()

    return PaimonConfig(
        critic_config=critic_config,
        rag_config=rag_config,
        llamp_config=llamp_config,
        web_search_config=web_search_config,
        **config,
    )
