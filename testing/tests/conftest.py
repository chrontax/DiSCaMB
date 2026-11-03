"""Shared fixtures for discamb container tests.

Each image builds pyDiSCaMB once: natively for the cpu backend, with SYCL for
GPU backends. ``run_in_env`` runs Python in that build; a SYCL build also runs
the CPU engines, which is how a test compares Engine4 with Engine3. Tests that
need the SYCL build are marked ``@pytest.mark.sycl`` and skipped elsewhere.
"""

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import pytest

WORKLOAD = Path(__file__).parent / "workloads" / "taam_fcalc.py"
WARMUP_STRUCTURE = "doc/data/tyrosine.pdb"


@dataclass(frozen=True)
class EnvRun:
    returncode: int
    stdout: str
    stderr: str
    wall_time_s: float


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "sycl: needs pyDiSCaMB built with SYCL (a GPU backend)"
    )


def pytest_collection_modifyitems(config, items):
    build = os.environ.get("DISCAMB_TEST_BUILD")
    if build == "sycl":
        return
    skip = pytest.mark.skip(reason=f"needs the sycl build, this image has {build!r}")
    for item in items:
        if "sycl" in item.keywords:
            item.add_marker(skip)


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
def build() -> str:
    return _setting("DISCAMB_TEST_BUILD")


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
    """Return ``run(script, args=(), timeout=None, check=True) -> EnvRun``.

    ``script`` is a path to a Python file or a string of Python code.
    """
    env_exec = _setting("DISCAMB_TEST_ENV_EXEC")

    def run(
        script: Union[str, Path],
        args: Sequence[str] = (),
        timeout: Optional[float] = None,
        check: bool = True,
    ) -> EnvRun:
        target = [str(script)] if isinstance(script, Path) else ["-c", script]
        start = time.perf_counter()
        proc = subprocess.run(
            [env_exec, "python", *target, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        result = EnvRun(
            proc.returncode, proc.stdout, proc.stderr, time.perf_counter() - start
        )
        if check and proc.returncode != 0:
            pytest.fail(f"workload exited with {proc.returncode}:\n{proc.stderr}")
        return result

    return run


@pytest.fixture(scope="session")
def sycl_warmup(run_in_env, source_dir):
    """Run Engine4 once on a tiny structure so later SYCL runs exclude kernel JIT.

    The CUDA driver compiles the PTX module on first load and keeps the result
    in its on-disk cache, so the compile cost is paid here, once per container.
    """
    run = run_in_env(
        WORKLOAD, args=[str(source_dir / WARMUP_STRUCTURE), "CPU_v4", "2.0"]
    )
    print(
        f"\n[sycl warm-up] Engine4 on tyrosine: {run.wall_time_s:.1f}s (includes kernel JIT)"
    )
