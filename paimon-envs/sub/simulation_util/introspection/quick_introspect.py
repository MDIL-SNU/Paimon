"""Core implementation of quick introspection utilities.

Capabilities:
 - Import diagnostics and suggestions
 - Class/method fuzzy discovery and signatures
 - Top-level function fuzzy discovery and signatures

Design notes:
 - Jedi-based static scanning is preferred first to avoid import side-effects
 - Falls back to safe runtime imports and `inspect` when static analysis yields nothing
 - Output is assembled into a textual report
"""

from __future__ import annotations

import ast
import difflib
import importlib
import importlib.util
import inspect
import io
import os
import pkgutil
import sys
from types import ModuleType
from typing import Any, Iterable

try:
    import jedi
except Exception:
    jedi = None  # type: ignore[assignment]


def _debug_engine(msg: str) -> None:
    try:
        if os.getenv("QI_DEBUG_ENGINE", "0") == "1":
            print(f"ENGINE: {msg}")
    except Exception:
        pass


def normalize(name: str) -> str:
    return (name or "").replace("_", "").lower()


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(a=normalize(a), b=normalize(b)).ratio()


def _silence_stdio_context():
    class _Silencer:
        def __enter__(self):
            self._old_out, self._old_err = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
            return self

        def __exit__(self, exc_type, exc, tb):
            sys.stdout, sys.stderr = self._old_out, self._old_err

    return _Silencer()


def safe_import(module_name: str) -> ModuleType | None:
    try:
        with _silence_stdio_context():
            return importlib.import_module(module_name)
    except Exception:
        return None


def iter_submodules(pkg: ModuleType, limit: int = 300) -> Iterable[str]:
    if not hasattr(pkg, "__path__"):
        return []
    count = 0
    try:
        for _finder, modname, _ispkg in pkgutil.walk_packages(
            pkg.__path__, prefix=pkg.__name__ + ".", onerror=lambda _name: None
        ):
            yield modname
            count += 1
            if count >= limit:
                break
    except Exception:
        return []


def list_attrs(module: ModuleType) -> dict[str, Any]:
    try:
        return {n: getattr(module, n) for n in dir(module)}
    except Exception:
        return {}


def is_public(name: str) -> bool:
    return bool(name) and not name.startswith("_")


def rank_names(cands: list[str], target: str) -> list[str]:
    t = normalize(target)
    return sorted(
        cands,
        key=lambda n: (
            normalize(n) != t,
            0 if normalize(n).startswith(t) else 1,
            -similarity(n, target),
            len(n),
        ),
    )


def rank_pairs(
    pairs: list[tuple[str, str]], target: str
) -> list[tuple[str, str]]:
    return sorted(
        pairs,
        key=lambda p: (
            normalize(p[0]) != normalize(target),
            0 if normalize(p[0]).startswith(normalize(target)) else 1,
            -similarity(p[0], target),
            len(p[1]),
        ),
    )


def extract_imports(code: str) -> list[tuple[str, list[str]]]:
    out = []
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            names = [a.name for a in node.names if a.name != "*"]
            if mod:
                out.append((mod, names))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                out.append((alias.name, []))
    return out


def top_repo_of(module_path: str) -> str:
    return (module_path or "").split(".")[0]


def module_belongs_to_root(mod: ModuleType, package_root: str | None) -> bool:
    if not package_root:
        return True
    try:
        root = os.path.abspath(package_root)
        if hasattr(mod, "__file__") and mod.__file__:
            return os.path.abspath(mod.__file__).startswith(root)
        if hasattr(mod, "__path__") and mod.__path__:
            for p in mod.__path__:
                if os.path.abspath(p).startswith(root):
                    return True
            return False
    except Exception:
        return True
    return True


def _get_repo_roots(repo_name: str, pkg_root: str | None) -> list[str]:
    if pkg_root and os.path.isdir(pkg_root):
        return [pkg_root]
    try:
        spec = importlib.util.find_spec(repo_name)
    except Exception:
        spec = None
    roots: list[str] = []
    if spec is not None:
        if getattr(spec, "submodule_search_locations", None):
            roots.extend(list(spec.submodule_search_locations))
        elif getattr(spec, "origin", None):
            roots.append(os.path.dirname(spec.origin))
    return [r for r in roots if r and os.path.isdir(r)]


def _module_name_from_path(file_path: str, roots: list[str]) -> str | None:
    file_path = os.path.abspath(file_path)
    for root in roots:
        root_abs = os.path.abspath(root)
        if file_path.startswith(root_abs):
            rel = os.path.relpath(file_path, root_abs)
            if rel.endswith("__init__.py"):
                rel = os.path.dirname(rel)
            mod = rel[:-3] if rel.endswith(".py") else rel
            mod = mod.replace(os.sep, ".")
            top_pkg = os.path.basename(os.path.normpath(root_abs))
            if mod:
                return f"{top_pkg}.{mod}" if mod else top_pkg
            return top_pkg
    return None


def search_symbol_in_package(
    repo: str, symbol: str, package_root: str | None = None
) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []

    def static_search_with_jedi() -> list[tuple[str, str]]:
        if jedi is None:
            return []
        roots = _get_repo_roots(repo, package_root)
        if not roots:
            return []
        matches: list[tuple[str, str]] = []
        max_files = 400
        seen_files = 0
        for root in roots:
            for dirpath, _dirs, files in os.walk(root):
                for fn in files:
                    if not fn.endswith(".py"):
                        continue
                    file_path = os.path.join(dirpath, fn)
                    try:
                        names = jedi.api.names(
                            path=file_path,
                            all_scopes=True,
                            definitions=True,
                            references=False,
                        )
                    except Exception:
                        names = []
                    for nm in names:
                        try:
                            typ = getattr(nm, "type", "")
                            nm_str = getattr(nm, "name", "")
                            if typ not in ("class", "function"):
                                continue
                            if not nm_str or not is_public(nm_str):
                                continue
                            is_match = (
                                nm_str == symbol
                                or normalize(symbol) in normalize(nm_str)
                                or similarity(nm_str, symbol) > 0.7
                            )
                            if not is_match:
                                continue
                            modname = None
                            full_name = getattr(nm, "full_name", None)
                            if isinstance(full_name, str) and "." in full_name:
                                modname = ".".join(full_name.split(".")[:-1])
                            if not modname:
                                modname = _module_name_from_path(file_path, roots)
                            if modname:
                                matches.append((nm_str, modname))
                        except Exception:
                            continue
                    seen_files += 1
                    if seen_files >= max_files:
                        break
                if seen_files >= max_files:
                    break
        dedup: list[tuple[str, str]] = []
        seen: set = set()
        for n, m in rank_pairs(matches, symbol):
            if (n, m) not in seen:
                seen.add((n, m))
                dedup.append((n, m))
        return dedup[:30]

    results = static_search_with_jedi()
    if results:
        _debug_engine("class/function search: jedi")
        return results

    pkg = safe_import(repo)
    if not pkg:
        return []
    attrs = list_attrs(pkg)
    for n, obj in attrs.items():
        is_match = (
            is_public(n)
            and (
                n == symbol
                or normalize(symbol) in normalize(n)
                or similarity(n, symbol) > 0.7
            )
        )
        if is_match and (inspect.isclass(obj) or inspect.isfunction(obj)):
            results.append((n, pkg.__name__))

    submods = list(iter_submodules(pkg, limit=400))
    sym_norm = normalize(symbol)
    submods_ranked = sorted(
        submods,
        key=lambda m: (sym_norm not in m.replace(".", "").lower(), len(m)),
    )
    for modname in submods_ranked:
        mod = safe_import(modname)
        if not mod or not module_belongs_to_root(mod, package_root):
            continue
        attrs = list_attrs(mod)
        for n, obj in attrs.items():
            if not is_public(n):
                continue
            is_match = (
                n == symbol
                or normalize(symbol) in normalize(n)
                or similarity(n, symbol) > 0.7
            )
            if is_match and (inspect.isclass(obj) or inspect.isfunction(obj)):
                results.append((n, mod.__name__))

    seen = set()
    ranked = []
    for n, m in rank_pairs(results, symbol):
        if (n, m) not in seen:
            seen.add((n, m))
            ranked.append((n, m))
    return ranked[:30]


def search_functions_in_package(
    repo: str,
    func_fragment: str,
    module_hint: str | None = None,
    package_root: str | None = None,
) -> list[tuple[str, str, str]]:
    results: list[tuple[str, str, str]] = []

    def static_collect_with_jedi() -> list[tuple[str, str, str]]:
        if jedi is None:
            return []
        roots = _get_repo_roots(repo, package_root)
        if not roots:
            return []
        collected: list[tuple[str, str, str]] = []
        max_files = 400
        seen_files = 0
        mod_hint_norm = normalize(module_hint or "")
        for root in roots:
            for dirpath, _dirs, files in os.walk(root):
                for fn in files:
                    if not fn.endswith(".py"):
                        continue
                    file_path = os.path.join(dirpath, fn)
                    try:
                        names = jedi.api.names(
                            path=file_path,
                            all_scopes=False,
                            definitions=True,
                            references=False,
                        )
                    except Exception:
                        names = []
                    for nm in names:
                        try:
                            if getattr(nm, "type", "") != "function":
                                continue
                            func_name = getattr(nm, "name", "")
                            if not is_public(func_name):
                                continue
                            is_match = (
                                func_name == func_fragment
                                or normalize(func_fragment) in normalize(func_name)
                                or similarity(func_name, func_fragment) > 0.7
                            )
                            if not is_match:
                                continue
                            modname = None
                            full_name = getattr(nm, "full_name", None)
                            if isinstance(full_name, str) and "." in full_name:
                                modname = ".".join(full_name.split(".")[:-1])
                            if not modname:
                                modname = _module_name_from_path(file_path, roots)
                            if not modname:
                                continue
                            if module_hint and mod_hint_norm not in normalize(modname):
                                continue
                            collected.append(
                                (f"from {modname} import {func_name}", func_name, "(...)")
                            )
                        except Exception:
                            continue
                    seen_files += 1
                    if seen_files >= max_files:
                        break
                if seen_files >= max_files:
                    break
        return collected

    static_res = static_collect_with_jedi()
    if static_res:
        _debug_engine("function search: jedi")
        return static_res

    def collect_from_module(mod: ModuleType):
        attrs = list_attrs(mod)
        for n, obj in attrs.items():
            if not is_public(n):
                continue
            if inspect.isfunction(obj):
                is_match = (
                    n == func_fragment
                    or normalize(func_fragment) in normalize(n)
                    or similarity(n, func_fragment) > 0.7
                )
                if is_match:
                    try:
                        sig = str(inspect.signature(obj))
                    except Exception:
                        sig = "(...)"
                    results.append((f"from {mod.__name__} import {n}", n, sig))

    if module_hint:
        mod = safe_import(module_hint)
        if mod and module_belongs_to_root(mod, package_root):
            collect_from_module(mod)
            if hasattr(mod, "__path__"):
                for sub in iter_submodules(mod, limit=400):
                    subm = safe_import(sub)
                    if subm and module_belongs_to_root(subm, package_root):
                        collect_from_module(subm)
            return results
        pkg = safe_import(repo)
        if pkg:
            frag = module_hint.replace("_", "").replace(".", "").lower()
            candidates = []
            for sub in iter_submodules(pkg, limit=400):
                key = sub.replace("_", "").replace(".", "").lower()
                if frag in key:
                    candidates.append(sub)
            base_token = module_hint.split(".")[-1]
            candidates = sorted(
                set(candidates),
                key=lambda s: (
                    s.find(base_token) if base_token in s else 9999,
                    len(s),
                ),
            )[:20]
            for cand in candidates:
                subm = safe_import(cand)
                if subm and module_belongs_to_root(subm, package_root):
                    collect_from_module(subm)
            if not results:
                norm_hint = frag
                sim_pool = []
                for sub in iter_submodules(pkg, limit=400):
                    norm_sub = sub.replace("_", "").replace(".", "").lower()
                    score = difflib.SequenceMatcher(a=norm_hint, b=norm_sub).ratio()
                    if score >= 0.7:
                        sim_pool.append((score, sub))
                for _score, cand in sorted(
                    sim_pool, key=lambda t: (-t[0], len(t[1]))
                )[:20]:
                    subm = safe_import(cand)
                    if subm and module_belongs_to_root(subm, package_root):
                        collect_from_module(subm)
        return results

    pkg = safe_import(repo)
    if not pkg:
        return results
    collect_from_module(pkg)
    for sub in iter_submodules(pkg, limit=400):
        subm = safe_import(sub)
        if subm and module_belongs_to_root(subm, package_root):
            collect_from_module(subm)
    return results


def _extract_methods_from_file(
    file_path: str, cls_name: str
) -> list[tuple[str, str]]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        tree = ast.parse(content)
    except Exception:
        return []
    methods: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for item in node.body:
                if isinstance(
                    item, (ast.FunctionDef, ast.AsyncFunctionDef)
                ) and is_public(item.name):
                    try:
                        args = []
                        for a in getattr(item.args, "posonlyargs", []) or []:
                            args.append(a.arg)
                        for a in item.args.args:
                            if a.arg != "self":
                                args.append(a.arg)
                        if item.args.vararg:
                            args.append("*" + item.args.vararg.arg)
                        for a in item.args.kwonlyargs:
                            args.append(a.arg)
                        if item.args.kwarg:
                            args.append("**" + item.args.kwarg.arg)
                        sig = "(" + ", ".join(args) + ")"
                    except Exception:
                        sig = "(...)"
                    methods.append((item.name, sig))
            break
    return methods


def method_suggestions(
    repo: str,
    class_hint: str,
    method_hint: str | None,
    package_root: str | None = None,
) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []

    def static_collect_with_jedi() -> list[tuple[str, str, str]]:
        if jedi is None:
            return []
        roots = _get_repo_roots(repo, package_root)
        if not roots:
            return []
        collected: list[tuple[str, str, str]] = []
        max_files = 400
        seen_files = 0
        for root in roots:
            for dirpath, _dirs, files in os.walk(root):
                for fn in files:
                    if not fn.endswith(".py"):
                        continue
                    file_path = os.path.join(dirpath, fn)
                    try:
                        names = jedi.api.names(
                            path=file_path,
                            all_scopes=True,
                            definitions=True,
                            references=False,
                        )
                    except Exception:
                        names = []
                    for nm in names:
                        try:
                            if getattr(nm, "type", "") != "class":
                                continue
                            cls_name = getattr(nm, "name", "")
                            if not is_public(cls_name):
                                continue
                            is_match = (
                                cls_name == class_hint
                                or normalize(class_hint) in normalize(cls_name)
                                or similarity(cls_name, class_hint) > 0.7
                            )
                            if not is_match:
                                continue
                            modname = None
                            full_name = getattr(nm, "full_name", None)
                            if isinstance(full_name, str) and "." in full_name:
                                modname = ".".join(full_name.split(".")[:-1])
                            if not modname:
                                modname = _module_name_from_path(file_path, roots)
                            if not modname:
                                continue
                            methods = _extract_methods_from_file(file_path, cls_name)
                            method_names = [
                                m
                                for m in methods
                                if not method_hint
                                or (
                                    m[0] == method_hint
                                    or normalize(method_hint) in normalize(m[0])
                                    or similarity(m[0], method_hint) > 0.7
                                )
                            ]
                            if method_hint:
                                ranked = sorted(
                                    method_names,
                                    key=lambda x: (
                                        normalize(x[0]) != normalize(method_hint),
                                        -similarity(x[0], method_hint),
                                        len(x[0]),
                                    ),
                                )[:8]
                            else:
                                ranked = sorted(method_names, key=lambda x: x[0])[:20]
                            for mname, sig in ranked:
                                collected.append(
                                    (f"from {modname} import {cls_name}", mname, sig)
                                )
                        except Exception:
                            continue
                    seen_files += 1
                    if seen_files >= max_files:
                        break
                if seen_files >= max_files:
                    break
        return collected[:20]

    static_res = static_collect_with_jedi()
    if static_res:
        _debug_engine("method suggestions: jedi")
        return static_res

    class_cands = search_symbol_in_package(repo, class_hint, package_root=package_root)
    pruned = []
    for sym, modname in class_cands:
        mod = safe_import(modname)
        if not mod:
            continue
        obj = getattr(mod, sym, None)
        if inspect.isclass(obj):
            pruned.append((sym, modname, obj))
    for sym, modname, cls in pruned[:10]:
        methods = {
            n: f
            for n, f in inspect.getmembers(cls, predicate=inspect.isfunction)
            if is_public(n)
        }
        if method_hint:
            ranked = rank_names(list(methods.keys()), method_hint)[:8]
        else:
            ranked = sorted(methods.keys())
        for n in ranked:
            try:
                sig = str(inspect.signature(methods[n]))
            except Exception:
                sig = "(...)"
            out.append((f"from {modname} import {sym}", n, sig))
    return out[:20]


def method_suggestions_across_repo(
    repo: str, method_hint: str, package_root: str | None = None
) -> list[tuple[str, str, str]]:
    results: list[tuple[str, str, str]] = []

    def extract_methods_from_file_all(file_path: str) -> list[tuple[str, str, str]]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            tree = ast.parse(content)
        except Exception:
            return []
        triples: list[tuple[str, str, str]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and is_public(node.name):
                cls_name = node.name
                for item in node.body:
                    if isinstance(
                        item, (ast.FunctionDef, ast.AsyncFunctionDef)
                    ) and is_public(item.name):
                        is_match = (
                            item.name == method_hint
                            or normalize(method_hint) in normalize(item.name)
                            or similarity(item.name, method_hint) > 0.7
                        )
                        if is_match:
                            try:
                                args = []
                                for a in getattr(item.args, "posonlyargs", []) or []:
                                    args.append(a.arg)
                                for a in item.args.args:
                                    if a.arg != "self":
                                        args.append(a.arg)
                                if item.args.vararg:
                                    args.append("*" + item.args.vararg.arg)
                                for a in item.args.kwonlyargs:
                                    args.append(a.arg)
                                if item.args.kwarg:
                                    args.append("**" + item.args.kwarg.arg)
                                sig = "(" + ", ".join(args) + ")"
                            except Exception:
                                sig = "(...)"
                            triples.append((cls_name, item.name, sig))
        return triples

    if jedi is not None:
        roots = _get_repo_roots(repo, package_root)
        if roots:
            max_files = 400
            seen_files = 0
            for root in roots:
                for dirpath, _dirs, files in os.walk(root):
                    for fn in files:
                        if not fn.endswith(".py"):
                            continue
                        file_path = os.path.join(dirpath, fn)
                        triples = extract_methods_from_file_all(file_path)
                        if triples:
                            modname = _module_name_from_path(file_path, roots)
                            if modname:
                                for cls_name, mname, sig in triples:
                                    results.append(
                                        (f"from {modname} import {cls_name}", mname, sig)
                                    )
                        seen_files += 1
                        if seen_files >= max_files:
                            break
                    if seen_files >= max_files:
                        break
            if results:
                _debug_engine("method suggestions across repo: jedi")
                return results

    pkg = safe_import(repo)
    if not pkg:
        return results

    def collect_from_module(mod: ModuleType):
        try:
            for name, obj in inspect.getmembers(mod, inspect.isclass):
                if not is_public(name):
                    continue
                cls = obj
                methods = {
                    n: f
                    for n, f in inspect.getmembers(cls, predicate=inspect.isfunction)
                    if is_public(n)
                }
                for mname, func in methods.items():
                    is_match = (
                        mname == method_hint
                        or normalize(method_hint) in normalize(mname)
                        or similarity(mname, method_hint) > 0.7
                    )
                    if is_match:
                        try:
                            sig = str(inspect.signature(func))
                        except Exception:
                            sig = "(...)"
                        results.append(
                            (f"from {mod.__name__} import {name}", mname, sig)
                        )
        except Exception:
            pass

    collect_from_module(pkg)
    for sub in iter_submodules(pkg, limit=400):
        subm = safe_import(sub)
        if subm and module_belongs_to_root(subm, package_root):
            collect_from_module(subm)
    return results


def suggest_import_fixes(
    module: str, names: list[str], package_root: str | None = None
) -> dict[str | None, list[str]]:
    suggestions: dict[str | None, list[str]] = {}
    repo = top_repo_of(module)

    mod = safe_import(module)
    for name in names or [None]:
        if name is None:
            suggestions[module] = [
                f"import {module}  # {'OK' if mod else 'ImportError'}"
            ]
            continue

        cands: list[str] = []
        if mod and hasattr(mod, name):
            cands.append(f"from {module} import {name}  # OK")
        else:
            syms = search_symbol_in_package(repo, name, package_root=package_root)
            for sym, modname in syms:
                cands.append(f"from {modname} import {sym}")
            if mod:
                names_in_mod = [n for n in dir(mod) if is_public(n)]
                for n in rank_names(names_in_mod, name)[:5]:
                    if hasattr(mod, n):
                        cands.append(f"from {module} import {n}")
        uniq = []
        seen = set()
        for s in cands:
            if s not in seen:
                seen.add(s)
                uniq.append(s)
        suggestions[name] = uniq[:10]
    return suggestions


def run_quick_introspect(
    *,
    code_content: str | None = None,
    class_hint: str | None = None,
    method_hint: str | None = None,
    package_path: str | None = None,
    function_hint: str | None = None,
    module_hint: str | None = None,
    repo_hint: str | None = None,
    max_suggestions: int | None = None,
    no_imports: bool = False,
) -> tuple[str, bool]:
    """Execute quick introspection and return (report, found_any).

    Returns:
        Tuple of (human-readable report string, whether any results were found).

    Parameter relationships (enforced):
    - repo_hint vs package_path: mutually exclusive; provide at most one
    - module_hint requires function_hint
    - If any of class_hint/method_hint is provided, one of repo_hint or package_path is required
    - If function_hint is provided, one of repo_hint or package_path is required

    Notes:
    - repo_hint is the top-level import module name (may differ from pip distribution name)
    - package_path must be an absolute path to the package root directory
    - method_hint can be provided without class_hint to trigger a repo-wide search (noisy)
    - code_content is OPTIONAL unless you want import diagnostics
    """
    # Validate relationships
    if repo_hint and package_path:
        raise ValueError("Provide only one of repo_hint or package_path.")
    if module_hint and not function_hint:
        raise ValueError("module_hint must be used together with function_hint.")

    # Prepare package root
    package_root: str | None = None
    package_name_from_path: str | None = None
    if package_path:
        candidate = os.path.abspath(package_path)
        if os.path.isdir(candidate):
            package_root = candidate
            parent = os.path.dirname(candidate)
            if parent not in sys.path:
                sys.path.insert(0, parent)
            package_name_from_path = os.path.basename(candidate.rstrip(os.sep))
        else:
            raise ValueError(
                f"package_path does not exist or is not a directory: {package_path}"
            )

    # Validate hints require repo context
    any_hint = bool(class_hint or method_hint or function_hint)
    if any_hint and repo_hint and not safe_import(repo_hint):
        try:
            import site as _site
            import sysconfig as _sc

            _pure = _sc.get_paths().get("purelib") or ""
            _sites = []
            try:
                _sites = _site.getsitepackages() or []
            except Exception:
                _sites = []
            _hint = _pure or (_sites[0] if _sites else "<site-packages>")
        except Exception:
            _hint = "<site-packages>"
        raise ValueError(
            f"repo_hint could not be imported. Provide package_path instead "
            f"(absolute path, or relative path starting from {_hint})."
        )

    if (class_hint or method_hint) and not (repo_hint or package_root):
        raise ValueError(
            "For class/method search, provide repo_hint or package_path."
        )

    if function_hint and not (repo_hint or package_root):
        raise ValueError(
            "For function search, provide repo_hint or package_path."
        )

    def limit_list(xs: list[Any]) -> list[Any]:
        return xs if max_suggestions is None else xs[: max(0, max_suggestions)]

    buffer = io.StringIO()
    found_any = False  # track whether any search found results

    def _writeln(line: str = ""):
        buffer.write(line + "\n")

    code: str = code_content or ""

    # If no code and no symbol hints, return guidance
    any_hint = bool(class_hint or method_hint or function_hint)
    if not code.strip() and not any_hint:
        _writeln("No sufficient parameters provided for quick introspection.")
        _writeln("\nHOW TO USE (provide parameters based on your error message):")
        _writeln("- Import errors: pass code_content to enable import diagnostics.")
        _writeln("- Class issues: provide class_hint and repo_hint (or package_path).")
        _writeln("- Method issues: provide method_hint and repo_hint; preferably also class_hint.")
        _writeln("- Function issues: provide function_hint and repo_hint; optionally module_hint.")
        return buffer.getvalue(), False

    # Parse imports
    imports: list[tuple[str, list[str]]] = []
    if code.strip() and not no_imports:
        try:
            imports = extract_imports(code)
        except Exception:
            _writeln("AST_PARSE_ERROR")
            raise

    # Import diagnostics
    if imports and not no_imports:
        found_any = True
        _writeln("=== Import Check & Suggestions ===")
        for module, names in imports:
            repo = top_repo_of(module)
            ok_mod = bool(safe_import(module))
            ok_repo = bool(safe_import(repo)) if repo else False
            _writeln(
                f"\n[Module] {module}  (repo={repo})  -> {'OK' if ok_mod else 'ImportError'}"
            )
            sugg = suggest_import_fixes(module, names, package_root=package_root)
            for name, lines in sugg.items():
                if name is None:
                    continue
                status = "OK" if any(l.endswith(" # OK") for l in lines) else "FIX"
                _writeln(f"  - Symbol: {name}  [{status}]")
                for i, line in enumerate(limit_list(lines), 1):
                    _writeln(f"      {i}. {line}")
            if not ok_mod:
                if not ok_repo:
                    _writeln("  TIP: The top-level import root may be incorrect.")
                else:
                    _writeln("  NOTE: Top-level import exists; submodule path is likely incorrect.")
    elif code.strip() and not imports and not no_imports:
        _writeln(
            "No imports found in the provided code. "
            "Add a line like: from pkg.sub import Symbol"
        )
    elif not code.strip() and not no_imports:
        _writeln(
            "Import diagnostics skipped: no code provided. "
            "Provide code_content to analyze imports."
        )

    # Method suggestions with class context
    if class_hint:
        if repo_hint:
            repo = repo_hint
        elif package_name_from_path:
            repo = package_name_from_path
        elif imports:
            repo = top_repo_of(imports[0][0])
        else:
            repo = top_repo_of(class_hint)
        _writeln(
            f"\n=== Method Suggestions (repo={repo}, class~{class_hint}, method~{method_hint or ''}) ==="
        )
        results = method_suggestions(repo, class_hint, method_hint, package_root=package_root)
        if results:
            found_any = True
            for imp, mname, sig in limit_list(results):
                _writeln(f"  {imp}    # method: {mname}{sig}")
        else:
            _writeln(
                f"  No class matching '{class_hint}' found in {repo}."
            )

    # Repo-wide method search
    if method_hint and not class_hint and (repo_hint or package_root):
        if repo_hint:
            repo = repo_hint
        elif package_name_from_path:
            repo = package_name_from_path
        elif imports:
            repo = top_repo_of(imports[0][0])
        else:
            repo = ""
        if repo:
            _writeln(
                f"\n=== Method Suggestions (repo={repo}, method~{method_hint}) - repo-wide search ==="
            )
            _writeln(
                "NOTE: --method_hint provided without --class_hint. "
                "Searching across all classes (may be noisy)."
            )
            widish = method_suggestions_across_repo(repo, method_hint, package_root=package_root)
            if widish:
                found_any = True
                for imp, mname, sig in limit_list(widish):
                    _writeln(f"  {imp}    # method: {mname}{sig}")
            else:
                _writeln(
                    f"  No methods matching '{method_hint}' found in {repo}."
                )

    # Function suggestions
    if function_hint:
        if repo_hint:
            repo = repo_hint
        elif package_name_from_path:
            repo = package_name_from_path
        elif imports:
            repo = top_repo_of(imports[0][0])
        elif module_hint:
            repo = top_repo_of(module_hint)
        else:
            repo = ""
        _writeln(
            f"\n=== Function Suggestions (repo~{repo or 'unknown'}, module_hint~{module_hint or ''}, function~{function_hint}) ==="
        )
        fres = (
            search_functions_in_package(
                repo, function_hint, module_hint=module_hint, package_root=package_root
            )
            if repo
            else []
        )
        if not fres and module_hint:
            mod = safe_import(module_hint)
            if not mod:
                _writeln(
                    "NOTE: module_hint could not be identified. "
                    "Consider removing module_hint and rerunning with only function_hint."
                )
        if fres:
            found_any = True
            for imp, fname, sig in limit_list(fres):
                _writeln(f"  {imp}    # function: {fname}{sig}")
        else:
            _writeln(
                f"  No function matching '{function_hint}' found in {repo or 'unknown'}."
            )

    return buffer.getvalue(), found_any


__all__ = [
    "run_quick_introspect",
    "search_symbol_in_package",
    "search_functions_in_package",
    "method_suggestions",
]
