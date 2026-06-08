"""Paimon package base module. Execute runtime constant code"""

import llama_index.core

from paimon.config import load_config_from_yaml

cfg = load_config_from_yaml()

if cfg.use_arize_phoenix:
    llama_index.core.set_global_handler(
        "arize_phoenix", endpoint="https://llamatrace.com/v1/traces"
    )

