"""Lightweight runtime probing helpers for debugging KeyError/AttributeError.

Usage examples:

# Probe a mapping for a key; on KeyError prints available keys
value = try_get_key(mapping, "target_key")

# Probe an object attribute; on AttributeError prints accessible attributes
attr_value = try_get_attr(obj, "attribute_name")
"""

from __future__ import annotations

import difflib
import inspect
from typing import Any


def _iter_public_dir(obj: Any) -> list[str]:
    try:
        names = [n for n in dir(obj) if not n.startswith("_")]
        names.sort()
        return names
    except Exception:
        return []


def show_all_keys_or_attrs(obj: Any) -> None:
    """Print all accessible keys (for mappings) or public attributes (for objects)."""
    try:
        if isinstance(obj, dict):
            try:
                keys = sorted(obj.keys())
            except Exception:
                keys = list(obj.keys())
            print("DICT_KEYS:", keys)
            print("TYPE:", type(obj).__name__)
            return

        # pydantic v2 style models
        if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
            try:
                keys = sorted(list(getattr(obj, "model_dump")().keys()))
            except Exception:
                keys = []
            print("MODEL_KEYS:", keys)
            print("TYPE:", type(obj).__name__)
            return

        # generic object
        names = _iter_public_dir(obj)
        print("ATTRS:", names)
        print("TYPE:", type(obj).__name__)
        print(
            "HINT: Some attributes may contain nested attributes; "
            "probe them individually if needed."
        )
    except Exception as e:
        print("PROBE_FAIL:", e)


def try_get_key(mapping: Any, key: Any) -> Any:
    """Attempt mapping[key].

    On success: print a concise success note and return the value.
    On KeyError: print available keys for the mapping and re-raise.
    """
    try:
        value = mapping[key]
        try:
            if value is None:
                print(
                    f"[probe_key] Found key {key!r} but value is None; "
                    "verify the intended key or upstream logic."
                )
            else:
                print(
                    f"[probe_key] OK: key {key!r} is present "
                    f"(type={type(value).__name__})."
                )
        except Exception:
            pass
        return value
    except KeyError:
        print(f"KeyError: missing key -> {key!r}")
        show_all_keys_or_attrs(mapping)
        raise


def try_get_attr(obj: Any, name: str) -> Any:
    """Attempt to access an attribute.

    On success: print a concise success note and return the value.
    On AttributeError: print accessible attributes and suggest similar names.
    """
    try:
        value = getattr(obj, name)
        try:
            if value is None:
                print(
                    f"[probe_attr] Found attribute {name!r} but value is None; "
                    "verify the intended attribute or upstream logic."
                )
            else:
                print(
                    f"[probe_attr] OK: attribute {name!r} is present "
                    f"(type={type(value).__name__})."
                )
        except Exception:
            pass
        return value
    except AttributeError:
        print(f"AttributeError: missing attribute -> {name!r}")
        show_all_keys_or_attrs(obj)
        _suggest_similar_attrs(obj, name)
        raise


def _suggest_similar_attrs(obj: Any, name: str) -> None:
    """Suggest similar attribute names based on string similarity."""
    try:
        def _normalize(s: str) -> str:
            return s.replace("_", "").lower()

        def _similar(a: str, b: str) -> float:
            return difflib.SequenceMatcher(
                a=_normalize(a), b=_normalize(b)
            ).ratio()

        candidates: list[str] = []

        # Instance-visible names
        try:
            candidates.extend([n for n in dir(obj) if not n.startswith("_")])
        except Exception:
            pass

        # Class-level properties/descriptors across MRO
        try:
            for base in inspect.getmro(type(obj)):
                for n, desc in getattr(base, "__dict__", {}).items():
                    if n.startswith("_"):
                        continue
                    is_descriptor = (
                        isinstance(desc, property)
                        or inspect.isdatadescriptor(desc)
                        or inspect.ismethoddescriptor(desc)
                    )
                    if is_descriptor:
                        candidates.append(n)
        except Exception:
            pass

        # Rank and print top suggestions
        uniq = sorted(set(candidates))
        ranked = sorted(
            uniq,
            key=lambda n: (
                _normalize(n) != _normalize(name),
                0 if _normalize(n).startswith(_normalize(name)) else 1,
                -_similar(n, name),
                len(n),
            ),
        )
        top = ranked[:10]
        if top:
            print("SUGGEST_ATTRS (by name similarity):", top)
            print(
                "HINT: Try these directly or probe nested attributes if needed."
            )
    except Exception:
        pass
