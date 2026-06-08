"""Code introspection tools for debugging and API discovery.

Available in agent environment via: from simulation_util.introspection import ...
"""

from simulation_util.introspection.quick_introspect import run_quick_introspect
from simulation_util.introspection.runtime_probe import (
    try_get_key,
    try_get_attr,
    show_all_keys_or_attrs,
)

__all__ = [
    "run_quick_introspect",
    "try_get_key",
    "try_get_attr",
    "show_all_keys_or_attrs",
]
