"""Shared fixtures for discamb container tests.

Tests run under pytest in the discamb-sycl environment. ``run_in_env`` executes
Python code in either environment ("cpu" or "sycl"), which is how a test
compares the SYCL build against its CPU counterpart.
"""

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import pytest

ENVS = ("cpu", "sycl")


@dataclass(frozen=True)
class EnvRun:
    env: str
    returncode: int
    stdout: str
    stderr: str
    wall_time_s: float


class _LabeledStream:
    def __init__(self, stream, label: str) -> None:
        self._stream = stream
        self._label = label
        self._at_line_start = True
        self._written = False

    def write(self, text: str) -> int:
        parts = [] if self._written or not text else ["\n"]
        self._written = self._written or bool(text)
        for chunk in text.splitlines(keepends=True):
            if self._at_line_start:
                parts.append(self._label)
            parts.append(chunk)
            self._at_line_start = chunk.endswith("\n")
        self._stream.write("".join(parts))
        self._stream.flush()
        return len(text)

    def flush(self) -> None:
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


@pytest.fixture(autouse=True)
def _label_test_output(request, monkeypatch):
    label = f"[{request.node.name}] "
    monkeypatch.setattr(sys, "stdout", _LabeledStream(sys.stdout, label))
    monkeypatch.setattr(sys, "stderr", _LabeledStream(sys.stderr, label))


def _setting(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.fail(f"{name} is not set; run the tests through testing/test-linux.sh")
    return value


@pytest.fixture(scope="session")
def backend() -> str:
    return _setting("DISCAMB_TEST_BACKEND")


@pytest.fixture(scope="session")
def system() -> str:
    return _setting("DISCAMB_TEST_SYSTEM")


@pytest.fixture(scope="session")
def results_dir() -> Path:
    return Path(_setting("DISCAMB_TEST_RESULTS"))


@pytest.fixture(scope="session")
def source_dir() -> Path:
    return Path(_setting("DISCAMB_TEST_SOURCE"))


@pytest.fixture(scope="session")
def run_in_env():
    """Return ``run(env, script, args=(), timeout=None, check=True) -> EnvRun``.

    ``script`` is a path to a Python file or a string of Python code.
    """
    env_exec = _setting("DISCAMB_TEST_ENV_EXEC")

    def run(
        env: str,
        script: Union[str, Path],
        args: Sequence[str] = (),
        timeout: Optional[float] = None,
        check: bool = True,
    ) -> EnvRun:
        if env not in ENVS:
            raise ValueError(f"unknown env {env!r}, expected one of {ENVS}")
        target = [str(script)] if isinstance(script, Path) else ["-c", script]
        start = time.perf_counter()
        proc = subprocess.run(
            [env_exec, env, "python", *target, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        result = EnvRun(env, proc.returncode, proc.stdout, proc.stderr, time.perf_counter() - start)
        if check and proc.returncode != 0:
            pytest.fail(f"[{env}] exited with {proc.returncode}:\n{proc.stderr}")
        return result

    return run
