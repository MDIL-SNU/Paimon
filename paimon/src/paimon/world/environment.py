"""Agent interaction to its working directory

TODO: use known sandbox solutions
TODO: [IMPORTANT] async, non-blocking run of fabric
"""

import os
import os.path as osp
import asyncio
import json
import inspect
import tempfile
import random
import string
import shlex
from typing import overload, Literal, Any
from pathlib import Path, PurePosixPath
from datetime import datetime

from jupyter_client.connect import tunnel_to_kernel
from jupyter_client.asynchronous.client import AsyncKernelClient
import textwrap
from invoke.runners import Result
from invoke.exceptions import CommandTimedOut
from pydantic_core import to_jsonable_python

from paimon import cfg
from paimon.util.log import debug, debug_var, debug_assert
import paimon.world as world

_alphabet = string.ascii_lowercase + string.digits

MAX_BASH_OUTPUT_CHARS = 4000


def _id_gen():
    return "".join(random.choices(_alphabet, k=8))


def is_id_already_present(id: str) -> bool:
    conn = world.get_connection()
    path = f"/home/{cfg.paimon_user}/wd/__{id}"
    ret: Result = conn.run(f"test -d {path}", warn=True, hide=True)
    return ret.ok


def new_environment(id: str | None = None, **kwargs) -> str:
    """Create a new environment or load environment from existing.
    TODO: Rework.

    Parameters
    ----------
    id
        Give env_id, instead of generation. If there is no dir, create. If there is
        dir, load it.
    kwargs
        keyword arguemtns to Environment

    Returns
    -------
    env_id
    """
    if not id:  # Create a completely new env
        env = Environment(id=id, **kwargs)
        world.register_env(env)
        return env.id
    else:
        try:
            # Check id is already registered, if already, return id
            world.get_env(id)
            return id
        except KeyError:
            # id is given and not the id is not registered.
            # create a completely new env with given id, if does not exist as folder
            # if already exist as folder, couple the folder to this instance
            env = Environment(id=id, **kwargs)
            # Either, register
            world.register_env(env)
            return env.id


def unique_filepath(filepath: str) -> str:
    if not os.path.isfile(filepath):
        return filepath
    else:
        dirname = os.path.dirname(filepath)
        fname = os.path.basename(filepath)
        name, ext = os.path.splitext(fname)
        cnt = 0
        new_name = f"{name}{cnt}{ext}"
        new_path = os.path.join(dirname, new_name)
        while os.path.exists(new_path):
            cnt += 1
            new_name = f"{name}{cnt}{ext}"
        return new_path


def is_valid_unix_dirname(name: str) -> bool:
    if not name:
        return False
    if name in {".", ".."}:
        return False
    if "/" in name or "\0" in name:
        return False
    if len(name) > 255:
        return False
    return True


class Environment:
    """Define working directory of agent"""

    _total_envs: int = 0

    def __init__(
        self,
        id: str | None = None,
        paimon_envs_root: str = cfg.paimon_envs_root,
    ) -> None:
        """This method should not be used directly. Use 'new_environment'

        Parameters
        ----------
        name
            nickname of this enviroment
        paimon_envs_root
            path to paimon-envs root (e.g., ".local/Paimon/paimon-envs")
            If provided, uses multi-environment system
            If None, uses legacy single-env system with setup_script

        Returns
        -------
        None
        """
        self.user_name = cfg.paimon_user

        if not id:
            debug("[environment] id is not given. create a new env with new id")
            id = _id_gen()
        else:
            debug("[environment] id is given.")

        self._id = id
        self._conn = world.get_connection()

        # Multi-environment system: each env has its own setup.sh
        # e.g., ".local/Paimon/paimon-envs/envs/base/setup.sh"
        self._paimon_envs_root = paimon_envs_root
        # TODO: Smarter way to detect available envs
        self._venv_map = {
            "base": f"{paimon_envs_root}/envs/base/setup.sh",  # default
            "sevennet": f"{paimon_envs_root}/envs/sevennet/setup.sh",
            "mace": f"{paimon_envs_root}/envs/mace/setup.sh",
        }
        debug(f"[environment] Multi-env: root={paimon_envs_root}")

        self.envs = {
            "PAIMON_ID": self.id,  # not used yet
            "OMP_NUM_THREADS": "1",
        }
        if mp_api := os.environ.get("MP_API_KEY"):
            self.envs["MP_API_KEY"] = mp_api  # give my mp_api

        self._initialized = False
        self._wd = ""  # << this is inside __init__ (?)
        self._sub_wds = {}
        self._ipy_sessions = {}

        self._setup_check()
        self._initialize()
        self._destroyed = False

    @property
    def id(self) -> str:
        """Get id of this environment"""
        return self._id

    @property
    def wd(self) -> str:
        """Get absolute path to the working directory"""
        if self._wd is None:
            raise RuntimeError("wd is no longer available")
        return self._wd

    def get_current_setup_script(self, venv_name: str | None = None) -> str:
        """Get the setup script path for the current or specified environment.

        Parameters
        ----------
        venv_name
            Optional environment name. If None, uses current environment.

        Returns
        -------
        str
            Path to setup script relative to user home (e.g.,
            ".local/Paimon/paimon-envs/envs/base/setup.sh")

        Raises
        ------
        ValueError
            If environment name is unknown.
        """
        venv_name = venv_name or "base"
        setup_script = self._venv_map.get(venv_name)
        if not setup_script:
            raise ValueError(
                f"Unknown environment: {venv_name}. "
                f"Available: {list(self._venv_map.keys())}"
            )
        return setup_script

    def create_link_to_wd(
        self, wd_prefix: str | None = None, to: str | None = None
    ) -> None:
        """Create symbolic link to working directory."""
        if self.wd is None:
            raise ValueError("working directory not created")
        wd = self.wd if not wd_prefix else osp.join(wd_prefix, f"wd/__{self.id}")
        to = to or f"./{self.id}"
        debug(wd)
        debug(to)
        os.symlink(wd, to, target_is_directory=True)

    def _setup_check(self) -> None:
        """Dev function for check integrity of paimon usr"""
        debug(
            f"[_setup_check] checking all registered envs: {list(self._venv_map.keys())}"
        )

        # Check wd/ directory exists
        res = self._conn.run("ls wd", hide=True, warn=True)
        assert res.ok, "Working directory 'wd/' not found on remote."

        # Check all registered venvs
        missing_setup = []
        failed_checks = []
        for venv_name, setup_path in self._venv_map.items():
            # setup.sh must exist
            res = self._conn.run(f"test -f {setup_path}", hide=True, warn=True)
            if not res.ok:
                missing_setup.append(f"  {venv_name}: {setup_path}")
                continue

            # Run check.sh if it exists
            check_path = setup_path.replace("setup.sh", "check.sh")
            res = self._conn.run(f"test -f {check_path}", hide=True, warn=True)
            if not res.ok:
                continue  # no check.sh, skip

            res = self._conn.run(
                f"source {setup_path} && bash {check_path}", hide=True, warn=True
            )
            if not res.ok:
                stderr = res.stderr.strip()
                failed_checks.append(f"  {venv_name}: {stderr}")

        assert not missing_setup, (
            "Setup scripts missing for registered envs:\n" + "\n".join(missing_setup)
        )
        assert not failed_checks, "Environment checks failed:\n" + "\n".join(
            failed_checks
        )

    def _rebuild_sub_wds(self) -> None:
        debug("[environment] rebuild subwds")
        result = self._conn.run(
            f"find {self._wd} -mindepth 1 -maxdepth 1 -type d",
            hide=True,
            warn=True,
        )

        assert result.ok, f"sub_wd rebuild is failed: {result.stderr}"

        for line in result.stdout.splitlines():
            abs_path = line.strip()
            # HPC server is linux POSIX
            sub_wd = PurePosixPath(abs_path).name
            self._sub_wds[sub_wd] = abs_path
        debug(f"[environment] rebuilt subwds: {self._sub_wds}")

    def _initialize(self) -> None:
        assert not self._initialized
        # TODO: need to remove ALL permissions except files made by this enviroment
        wd = f"wd/__{self.id}"

        rebuild_sub_wd_flag = False
        if not is_id_already_present(self.id):
            # The wd is not exists => fresh init
            path = self._conn.run(
                (
                    f"mkdir {wd}"
                    f"&& chmod 777 {wd}"
                    f"&& cd {wd}"
                    f"&& echo {datetime.now()}: INIT > .history"
                    f"&& chmod 666 .history"
                    f"&& mkdir .trash_bin"
                    f"&& chmod 775 .trash_bin"
                    "; echo $PATH"
                ),
                hide=True,
            ).stdout.strip()
        else:
            path = self._conn.run(
                (
                    f"cd {wd}"
                    f"&& echo {datetime.now()}: RECONNECT >> .history"
                    "; echo $PATH"
                ),
                hide=True,
            ).stdout.strip()
            rebuild_sub_wd_flag = True
        # TODO: for function_signature tool but seems redundant
        self.envs.update({"PATH": f"{path}:/home/{self.user_name}/.local/bin/"})

        home = self._conn.run("pwd", hide=True).stdout.strip()
        self._wd = osp.join(home, f"wd/__{self.id}")
        self._wd_wo_home = f"wd/__{self.id}"

        self._trash_bin = ".trash_bin"
        abs_trash_bin = osp.join(self._wd, self._trash_bin)
        self._sub_wds[".trash_bin"] = abs_trash_bin

        if rebuild_sub_wd_flag:
            self._rebuild_sub_wds()

    def get_sub_wd_path(self, sub_wd: str, wo_home: bool = False) -> str:
        """Get absolute path to sub_wd. Create if not have.

        Parameters
        ----------
        sub_wd
            directory name. Should be valid for unix directory name

        Returns
        -------
        absolute path to the sub_wd
        """
        if not is_valid_unix_dirname(sub_wd):
            raise ValueError(f"{sub_wd} is not good for dir name")

        wd = self.wd if not wo_home else self._wd_wo_home
        abs_sub_wd = osp.join(wd, sub_wd)
        if sub_wd not in self._sub_wds:
            self._conn.run(f"mkdir {abs_sub_wd} && chmod 775 {abs_sub_wd}")
            self._sub_wds[sub_wd] = abs_sub_wd

        return abs_sub_wd

    async def discard_sub_wd(self, sub_wd: str, prefix: str) -> str:
        if sub_wd not in self._sub_wds:
            raise ValueError(f"The sub_wd: {sub_wd}, not exists in: {self._sub_wds}")
        self._sub_wds.pop(sub_wd)

        abs_sub_wd = osp.join(self.wd, sub_wd)

        prefix = shlex.quote(prefix)
        to_dir = self.sys_run(
            f'( i=0; while [ -d "{prefix}$i" ]; do ((i++)); done; mkdir "{prefix}$i"; echo "{prefix}$i" )',
            sub_wd=self._trash_bin,
        ).strip()
        self.sys_run(f"mv {abs_sub_wd} {to_dir}", sub_wd=self._trash_bin)

        try:
            if await self.ipy_is_alive(sub_wd):
                await self.ipy_stop(sub_wd)
        except Exception as e:
            debug(f"[ipy_start] {e}")
            pass
        self._ipy_sessions.pop(sub_wd, None)
        return to_dir

    def sys_run(self, command: str, sub_wd: str | None = None, **kwargs) -> str:
        """Alias for system run."""
        debug(f"[sys_run] sub_wd={sub_wd} command={command}")
        return self.run(
            command=command,
            wrap_for_llm=False,
            no_history=True,
            sub_wd=sub_wd,
            assert_failure=True,
            **kwargs,
        ).stdout

    def run_with_timeout_track(
        self,
        command: str,
        *,
        wd: str,
        timeout: float | None,
        **kwargs,
    ) -> Result:
        """
        Wrapper for self._conn.run() to manually abort process on timeout
        Assumption:
        - no pipeline
        - no multi-line command
        - no child process is spawned by the command
        """
        use_timeout_tracking = (
            timeout is not None
            and not kwargs.get("disown", False)
            and not kwargs.get("asynchronous", False)
        )
        if not use_timeout_tracking:
            return self._conn.run(
                command, hide=True, warn=True, env=self.envs, timeout=timeout, **kwargs
            )

        q_command = shlex.quote(command)
        timeout_s = f"{timeout}s"
        q_timeout_s = shlex.quote(timeout_s)
        timeout_cmd = (
            f"timeout --signal=TERM --kill-after=0.5s {q_timeout_s} "
            f"bash -c {q_command}"
        )

        res = self._conn.run(
            timeout_cmd, hide=True, warn=True, env=self.envs, timeout=None, **kwargs,
        )
        # timeout command returns 124 ($?) on timeout
        # we have to raise CommandTimedOut exception to match previous behavior
        if res.return_code == 124:
            raise CommandTimedOut(res, timeout=timeout)
        return res

    @overload
    def run(
        self,
        command: str,
        wrap_for_llm: Literal[False],
        sub_wd: str | None = None,
        no_history: bool = False,
        timeout: float | None = 60,
        assert_failure: bool = False,
        venv_name: str | None = None,
        **kwargs,
    ) -> Result: ...

    @overload
    def run(
        self,
        command: str,
        wrap_for_llm: Literal[True],
        sub_wd: str | None = None,
        no_history: bool = False,
        timeout: float | None = 60,
        assert_failure: bool = False,
        venv_name: str | None = None,
        **kwargs,
    ) -> str: ...

    def run(
        self,
        command: str,
        wrap_for_llm: bool = False,
        sub_wd: str | None = None,
        no_history: bool = False,
        timeout: float | None = 60,
        assert_failure: bool = False,
        venv_name: str | None = None,
        **kwargs,
    ) -> Result | str:
        """Run (should be) shell command

        Parameters
        ----------
        command
            bash command to execute
        wrap_for_llm
            if True, warp stdout and stderr into single string and returns it
        sub_wd
            if given, use sub working directry (create if not exists)
        no_history
            do not record .history for this command
        timeout
            timeout of command in seconds. default is 60.
        venv_name
            which virtual environment to use ("base", "sevennet", "mace", etc.)
            If None, uses agent's current_venv from state

        Returns
        -------
        Result
            A container for information about the result of a command execution.
        """
        assert not self._destroyed

        wd = self.wd if not sub_wd else self.get_sub_wd_path(sub_wd)

        if not no_history:
            caller = inspect.stack()[1].function
            cmd_str_meta = f"{datetime.now()} {sub_wd} {caller}: {command}"
            hist = osp.join(self.wd, ".history")

            # This works with cmd_str_meta having "<" or "EOF".
            # TODO: Other exceptions?
            with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
                tmp.write(cmd_str_meta)
                tmp.flush()
                tmp_path = tmp.name
            self._conn.get(remote=hist, local=tmp_path)
            with open(tmp_path, "a") as f:
                f.write(f"{cmd_str_meta}\n")
            self._conn.put(local=tmp_path, remote=hist)

        # Build command prefix with environment activation
        # Multi-environment system: each env's setup.sh handles everything
        debug(f"[run] using {venv_name}")
        venv_name = venv_name or "base"
        venv_setup = self._venv_map.get(venv_name)
        if not venv_setup:
            raise ValueError(
                f"Unknown environment: {venv_name}. Available: {list(self._venv_map.keys())}"
            )

        pre = f"source /home/{self.user_name}/{venv_setup} && cd {wd} "

        command = pre + "&& " + command

        res = self.run_with_timeout_track(command=command, wd=wd, timeout=timeout, **kwargs)
        debug_assert(
            not (assert_failure and res.return_code != 0),
            f"[run] cmd: {command}\nstdout: {res.stdout}\nstderr: {res.stderr}",
        )

        if wrap_for_llm:
            ret = []
            for x in ("stdout", "stderr"):
                val = getattr(res, x)
                if len(val) > MAX_BASH_OUTPUT_CHARS:
                    val = f"<system>outputs are truncated!</system>\n{val[:MAX_BASH_OUTPUT_CHARS]}"
                ret.append(
                    """\
<{}>
{}
</{}>
""".format(x, val.rstrip(), x)
                    if val
                    else "(no {})".format(x)
                )
            return "\n".join(ret)
        else:
            return res

    def read_json(self, remote_path: str, sub_wd: str | None = None) -> dict:
        wd = self.wd if not sub_wd else self.get_sub_wd_path(sub_wd)

        assert remote_path.endswith(".json")
        path = osp.join(wd, remote_path)
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            self._conn.get(path, local=tmp_path)
            with open(tmp_path, "r") as f:
                dct = json.load(f)
        finally:
            os.remove(tmp_path)
        return dct

    def read_file(self, remote_path: str, sub_wd: str | None = None) -> str:
        wd = self.wd if not sub_wd else self.get_sub_wd_path(sub_wd)
        path = osp.join(wd, remote_path)
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            self._conn.get(path, local=tmp_path)
            with open(tmp_path, "r") as f:
                content = f.read()
        finally:
            os.remove(tmp_path)
        return content

    def write_file(
        self, content: str, remote_path: str, sub_wd: str | None = None
    ) -> None:
        """Write file to the working directory

        Parameters
        ----------
        content
            file content
        remote_path
            relative path from working directory.
        sub_wd
            if given, use sub working directry (create if not exists)

        Raises
        ------
        ValueError
            when 'who' tries to overwrite someone else's file
        """
        wd = self.wd if not sub_wd else self.get_sub_wd_path(sub_wd)

        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            tmp.write(content)
            tmp.flush()
            tmp_path = tmp.name
        try:
            actual_path = osp.join(wd, remote_path)

            self._conn.put(local=tmp_path, remote=actual_path)
            self._conn.run(f"chmod 664 {actual_path}")
        finally:
            os.remove(tmp_path)

    def write_json(
        self,
        obj,
        filename: str,
        sub_wd: str | None = None,
        indent: int = 2,
        **kwargs,
    ) -> None:
        jsonable = to_jsonable_python(obj, **kwargs)
        # self.write_file(json.dumps(obj, indent=indent), filename, sub_wd=sub_wd)
        self.write_file(json.dumps(jsonable, indent=indent), filename, sub_wd=sub_wd)

    def file_exists(self, filename: str, sub_wd: str | None = None) -> bool:
        """Check whether given file exists and under working directory

        Parameters
        ----------
        filename
            name of file
        sub_wd
            if provided, assume working directory is osp.join(self.wd, sub_wd)

        Raises
        ------
        file exists
        """
        wd = self.wd if not sub_wd else self.get_sub_wd_path(sub_wd)

        # Prevent code injection
        filename = shlex.quote(filename)
        fpath = osp.join(wd, filename)

        ret: Result = self._conn.run(f"realpath -e {fpath}", hide=True, warn=True)
        if ret.return_code != 0:
            return False

        resolved_path = ret.stdout.strip()
        if osp.commonpath([resolved_path, wd]) != wd:
            return False

        return True

    def append_json(
        self,
        key: str,
        value,
        filename: str,
        sub_wd: str | None = None,
        indent: int = 2,
        **kwargs,
    ) -> None:
        """Append value to a list in JSON file under the specified key

        Parameters
        ----------
        key
            Top-level dictionary key
        value
            Value to append to the list
        filename
            JSON filename (must end with .json)
        sub_wd
            if given, use sub working directory
        indent
            JSON indentation level
        **kwargs
            Additional arguments for to_jsonable_python
        """
        assert filename.endswith(".json")

        if self.file_exists(filename, sub_wd=sub_wd):
            try:
                data = self.read_json(filename, sub_wd=sub_wd)
            except:  # TODO
                data = {}
        else:
            data = {}

        # Append to list or create new list
        if key in data:
            assert isinstance(data[key], list), (
                f"Key '{key}' exists but is not a list"
            )
            data[key].append(value)
        else:
            data[key] = [value]

        # Write back
        self.write_json(data, filename, sub_wd=sub_wd, indent=indent, **kwargs)

    def append_jsonl(
        self,
        entries: list[dict],
        filename: str,
        sub_wd: str | None = None,
    ) -> None:
        """Append JSON Lines entries to a file.

        Parameters
        ----------
        entries
            List of dicts, each written as one JSON line.
        filename
            Filename (must end with .jsonl).
        sub_wd
            If given, use sub working directory.
        """
        assert filename.endswith(".jsonl")
        lines = "\n".join(json.dumps(e) for e in entries) + "\n"

        wd = self.wd if not sub_wd else self.get_sub_wd_path(sub_wd)
        remote_path = osp.join(wd, filename)

        # Download existing content if file exists, then append
        existing = ""
        if self.file_exists(filename, sub_wd=sub_wd):
            with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                self._conn.get(remote_path, local=tmp_path)
                with open(tmp_path, "r") as f:
                    existing = f.read()
            finally:
                os.remove(tmp_path)

        self.write_file(existing + lines, filename, sub_wd=sub_wd)

    async def put(self, path: str, sub_wd: str | None = None) -> None:
        if not osp.isfile(path):
            raise FileNotFoundError
        wd = (
            self._wd_wo_home
            if not sub_wd
            else self.get_sub_wd_path(sub_wd, wo_home=True)
        )
        filename = osp.basename(path)
        debug(f"[env:put] {path} to {osp.join(wd, filename)}")
        self._conn.put(local=path, remote=osp.join(wd, filename))

    async def python_call(
        self,
        func_body: str,
        timeout: int = 60,
        sub_wd: str | None = None,
        func_kwargs: dict[str, Any] | None = None,
        venv_name: str | None = None,
    ) -> dict[str, Any] | str | list:
        """
        Parameters
        ----------
        func_body
            the body of a Python function (no def line, no indentation needed)
            It MUST return a JSON-serializable value and handle all errors internal
        timeout
            timeout
        sub_wd
            sub wd name to run the command
        func_kwargs
            Json serializable keyword argument to be sent to the func
        venv_name
            which virtual environment to use ("base", "sevennet", "mace", etc.)
            If None, uses agent's current_venv from state

        Example
            func_body = '''
                import os, json
                path = k["path"]
                return {"exists": os.path.exists(path)}
            '''
            rr = await env.call(func_body, path="/etc/hosts")
            print(rr["exists"])  # True
        """

        # - defines _func(*a, **k) with your body
        # - executes it with provided args/kwargs
        # - prints exactly one JSON line
        # - exits with non-zero on error
        script = (
            textwrap.dedent("""
            import json, sys, traceback
            def _func(*a, **k):
        """)
            + textwrap.indent(textwrap.dedent(func_body).rstrip() + "\n", "    ")
            + textwrap.dedent(f"""
            k = {json.dumps(func_kwargs)}
            out = _func(**k)
            print(json.dumps(out, ensure_ascii=False))
        """)  # noqa: E501
        )

        wd = self.wd if not sub_wd else self.get_sub_wd_path(sub_wd)

        # Build command prefix with environment activation
        debug(f"[python_call] using {venv_name}")
        # Multi-environment system: each env's setup.sh handles everything
        venv_name = venv_name or "base"
        venv_setup = self._venv_map.get(venv_name)
        if not venv_setup:
            raise ValueError(
                f"Unknown environment: {venv_name}. Available: {list(self._venv_map.keys())}"
            )

        pre = f"source /home/{self.user_name}/{venv_setup} && cd {wd} "

        cmd = "python3 - <<'END'\n" + script + "END"
        cmd = pre + "&& " + cmd
        r = self._conn.run(
            cmd,
            hide=True,
            warn=True,
            env=self.envs,
            timeout=timeout,
            pty=False,
            asynchronous=True,
        ).join()

        if r.exited != 0:
            raise RuntimeError(
                f"remote exited {r.exited}\n"
                f"stdout:\n{(r.stdout).strip()}"
                f"stderr:\n{(r.stderr).strip()}"
            )

        last_line = r.stdout.strip().splitlines()[-1] if r.stdout else ""
        try:
            payload = json.loads(last_line) if last_line else {}
        except json.JSONDecodeError as e:
            # Must not happen
            raise RuntimeError("Failed to parse return string into json dict") from e

        return payload

    def list_working_directory(self, sub_wd: str | None = None) -> str:
        wd = self.wd if not sub_wd else self.get_sub_wd_path(sub_wd)

        ret: Result = self._conn.run(f"ls -gGtrh {wd}", hide=True)

        if ret.return_code != 0:
            raise RuntimeError(f"Failed to list {wd!r}: {ret.stderr.strip()}")

        return ret.stdout

    def remove_wd(self):
        """Remove its working directory"""
        self._conn.run(f"ls {self.wd}/ && rm -r {self.wd}")
        self._destroyed = True
        self._wd = None

    async def ipy_start(
        self, sub_wd: str, *, wait_timeout: float = 8.0, venv_name: str | None = None
    ) -> None:
        """
        Start a background ipykernel for this sub_wd (one per sub_wd).
        if already running, does nothing.

        Parameters
        ----------
        sub_wd
            sub-working directory name
        wait_timeout
            timeout for waiting kernel to start
        venv_name
            which virtual environment to use ("base", "sevennet", "mace", etc.)
            If None, uses agent's current_venv from state
            IMPORTANT: IPykernel sessions are stateful - environment is set at start time
        """
        if sub_wd in self._ipy_sessions:
            try:
                if await self.ipy_is_alive(sub_wd):
                    return
            except Exception as e:
                debug(f"[ipy_start] {e}")
                pass
            # not alive or execption raised; drop it
            self._ipy_sessions.pop(sub_wd, None)

        remote_conn = f"/tmp/ipy_{self.id}_{sub_wd}.json"
        remote_pid = f"{remote_conn}.pid"
        remote_log = f"{remote_conn}.log"

        # launch ipykernel in the background, inheriting your env/cwd via self.run
        # note: disown to returns immediately; we persist PID for later cleanup
        launch = (
            f"nohup python -m ipykernel -f {remote_conn} "
            f"> {remote_log} 2>&1 & echo $! > {remote_pid}"
        )
        wd = self.wd if not sub_wd else self.get_sub_wd_path(sub_wd)

        # Build command prefix with environment activation
        debug(f"[ipy_start] using {venv_name}")
        # Multi-environment system: each env's setup.sh handles everything
        venv_name = venv_name or "base"
        venv_setup = self._venv_map.get(venv_name)
        if not venv_setup:
            raise ValueError(
                f"Unknown environment: {venv_name}. Available: {list(self._venv_map.keys())}"
            )

        pre = f"source /home/{self.user_name}/{venv_setup} && cd {wd} "

        cmd = pre + "&& " + launch
        self._conn.run(
            cmd,
            hide=True,
            warn=True,
            env=self.envs,
            timeout=None,
            pty=False,
            disown=True,
        )

        # poll until connection file is written (or timeout)
        deadline = asyncio.get_event_loop().time() + wait_timeout
        while True:
            chk = self.sys_run(
                f"test -s {remote_conn} && echo READY || true", sub_wd=sub_wd
            )
            if "READY" in chk:
                break
            if asyncio.get_event_loop().time() > deadline:
                raise TimeoutError(
                    f"ipykernel for {sub_wd} did not produce {remote_conn} in time"
                )
            await asyncio.sleep(0.1)

        # cache minimal session metadata
        self._ipy_sessions[sub_wd] = {
            "remote_conn": remote_conn,
            "remote_pid": remote_pid,
            "remote_log": remote_log,
            "client": None,  # filled lazily on first exec/connect
            "kern_conn_info": None,
        }

    async def ipy_connect(self, sub_wd: str):
        if sub_wd not in self._ipy_sessions:
            await self.ipy_start(sub_wd)

        sess = self._ipy_sessions[sub_wd]
        if sess.get("client") is not None:
            return sess["client"]

        # --- fetch remote connection JSON (stays the same) ---
        cat_stdout = self.sys_run(f"cat {sess['remote_conn']}")

        kern_conn_info = json.loads(cat_stdout)
        sess["kern_conn_info"] = kern_conn_info

        sshserver = f"{self._conn.user}@{self._conn.host}:{self._conn.port}"
        local_ports = tunnel_to_kernel(kern_conn_info, sshserver)

        local_kern_conn_info = kern_conn_info.copy()
        for i, port_name in enumerate(("shell", "iopub", "stdin", "hb", "control")):
            local_kern_conn_info[port_name + "_port"] = local_ports[i]

        client = AsyncKernelClient()
        client.load_connection_info(local_kern_conn_info)
        client.start_channels()
        await client.wait_for_ready(timeout=5.0)

        sess["client"] = client
        return client

    async def ipy_is_alive(self, sub_wd: str) -> bool:
        sess = getattr(self, "_ipy_sessions", {}).get(sub_wd)
        if not sess:
            return False
        pidf = sess["remote_pid"]
        ret = self.sys_run(
            f"test -s {pidf} && kill -0 $(cat {pidf}) >/dev/null 2>&1 && echo ALIVE || true",
            sub_wd=sub_wd,
        )
        return "ALIVE" in ret

    async def ipy_exec(
        self, sub_wd: str, code: str, *, timeout: float = 20.0
    ) -> str:
        client = await self.ipy_connect(sub_wd)

        _ = client.execute(
            code, allow_stdin=False, store_history=True, stop_on_error=False
        )
        stdout_parts, stderr_parts, result_text = [], [], None

        # collect messages until kernel goes idle
        while True:
            try:
                msg = await client.get_iopub_msg(timeout=timeout)
            except Exception:
                break
            mtype = msg["header"]["msg_type"]
            content = msg["content"]

            if mtype == "stream":
                if content.get("name") == "stderr":
                    stderr_parts.append(content.get("text", ""))
                else:
                    stdout_parts.append(content.get("text", ""))
            elif mtype in ("execute_result", "display_data"):
                data = content.get("data", {})
                # prefer plain text repr
                result_text = data.get("text/plain", result_text)
            elif mtype == "error":
                tb = "\n".join(content.get("traceback", []))
                stderr_parts.append(
                    f"{content.get('ename')}: {content.get('evalue')}\n{tb}"
                )
            elif mtype == "status" and content.get("execution_state") == "idle":
                break

        out = "".join(stdout_parts)
        err = "".join(stderr_parts)
        if result_text:
            # append a blank line + textual result, similar to a REPL echo
            if out and not out.endswith("\n"):
                out += "\n"
            out += ("\n" if out else "") + str(result_text)

        # mimic "text in/out": return combined, but keep stderr if you care to parse later
        return out + (("\n" + err) if err else "")

    async def ipy_stop(self, sub_wd: str) -> None:
        """
        Stop and clean up the ipykernel for sub_wd (if present).
        """
        sess = getattr(self, "_ipy_sessions", {}).get(sub_wd)
        if not sess:
            return

        # close client channels first (non-fatal if already closed)
        client: AsyncKernelClient | None = sess.get("client")
        if client is not None:
            try:
                client.stop_channels()
            except Exception:
                pass

        # kill remote process and remove files (pid/conn/log)
        pidf = sess["remote_pid"]
        connf = sess["remote_conn"]
        logf = sess["remote_log"]
        self.sys_run(
            f"if test -s {pidf}; then kill -9 $(cat {pidf}) >/dev/null 2>&1 || true; fi",
            sub_wd=sub_wd,
        )
        self.sys_run(f"rm -f {connf} {pidf} {logf}", sub_wd=sub_wd)

        # drop from registry
        self._ipy_sessions.pop(sub_wd, None)
