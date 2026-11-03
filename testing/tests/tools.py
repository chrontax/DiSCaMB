"""Diagnostic helpers importable from the discamb container tests.

``GpuMemoryProfiler`` samples GPU memory between ``start()`` and ``stop()`` with
the vendor tool of the backend it detects (cuda -> ``nvidia-smi``, sampled every
``DEFAULT_INTERVAL_MS`` milliseconds). ``start()`` takes a one-shot reading as
the baseline and fails there if the tool does not work; ``stop()`` returns a
``MemoryProfile`` holding the peak and the mean of the samples, in MiB.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Type

DEFAULT_INTERVAL_MS = 10
SAMPLER_STOP_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class MemoryProfile:
    backend: str
    peak_mib: float
    mean_mib: float
    baseline_mib: float
    samples: int
    duration_s: float

    @property
    def peak_above_baseline_mib(self) -> float:
        return self.peak_mib - self.baseline_mib


class _MemorySampler:
    backend = ""
    tool = ""

    def __init__(self, device: int, interval_ms: int) -> None:
        self.device = device
        self.interval_ms = interval_ms
        self.baseline = 0.0
        self._process: Optional[subprocess.Popen] = None
        self._reader: Optional[threading.Thread] = None
        self._stderr = None
        self._stderr_text = ""
        self._samples: List[float] = []
        self._rejected: List[str] = []

    @classmethod
    def available(cls) -> bool:
        return shutil.which(cls.tool) is not None

    def probe_command(self) -> List[str]:
        raise NotImplementedError

    def command(self) -> List[str]:
        raise NotImplementedError

    def parse(self, line: str) -> float:
        raise NotImplementedError

    def read_once(self) -> float:
        try:
            result = subprocess.run(
                self.probe_command(), capture_output=True, text=True
            )
        except FileNotFoundError:
            raise RuntimeError(f"{self.tool} is not available on PATH") from None
        for line in result.stdout.splitlines():
            if line.strip():
                try:
                    return self.parse(line.strip())
                except ValueError:
                    break
        raise RuntimeError(
            f"{self.tool} gave no memory reading (exit code {result.returncode}; "
            f"stdout: {result.stdout.strip()!r}; stderr: {result.stderr.strip()!r})"
        )

    def start(self) -> None:
        self.baseline = self.read_once()
        self._stderr = tempfile.TemporaryFile(mode="w+")
        self._process = subprocess.Popen(
            self.command(), stdout=subprocess.PIPE, stderr=self._stderr, text=True
        )
        self._reader = threading.Thread(target=self._collect, daemon=True)
        self._reader.start()

    def _collect(self) -> None:
        for line in self._process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self._samples.append(self.parse(line))
            except ValueError:
                if len(self._rejected) < 5:
                    self._rejected.append(line)

    def _shut_down(self) -> None:
        # SIGINT, not SIGTERM: the vendor tools treat it as end of loop and exit
        # normally, flushing samples still sitting in their stdout buffer.
        self._process.send_signal(signal.SIGINT)
        try:
            self._process.wait(timeout=SAMPLER_STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=SAMPLER_STOP_TIMEOUT_S)
        self._reader.join(timeout=SAMPLER_STOP_TIMEOUT_S)
        self._process.stdout.close()
        self._stderr.seek(0)
        self._stderr_text = self._stderr.read().strip()
        self._stderr.close()

    def _no_samples_message(self) -> str:
        details = [f"exit code {self._process.returncode}"]
        if self._stderr_text:
            details.append(f"stderr: {self._stderr_text}")
        if self._rejected:
            details.append(f"unparsed output: {self._rejected}")
        return (
            f"{self.tool} produced no memory samples ({'; '.join(details)}); "
            "the profiled window may be shorter than the tool's start-up time"
        )

    def stop(self) -> List[float]:
        self._shut_down()
        if not self._samples:
            raise RuntimeError(self._no_samples_message())
        return self._samples


class NvidiaSmiSampler(_MemorySampler):
    backend = "cuda"
    tool = "nvidia-smi"

    def probe_command(self) -> List[str]:
        return [
            self.tool,
            f"--id={self.device}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ]

    def command(self) -> List[str]:
        return [*self.probe_command(), f"--loop-ms={self.interval_ms}"]

    def parse(self, line: str) -> float:
        return float(line)


SAMPLERS: Tuple[Type[_MemorySampler], ...] = (NvidiaSmiSampler,)


def detect_backend() -> str:
    backend = os.environ.get("DISCAMB_TEST_BACKEND")
    if backend:
        return backend
    for sampler in SAMPLERS:
        if sampler.available():
            return sampler.backend
    tools = ", ".join(sampler.tool for sampler in SAMPLERS)
    raise RuntimeError(
        f"no GPU backend detected: none of these tools is on PATH: {tools}"
    )


def _sampler_type(backend: str) -> Type[_MemorySampler]:
    for sampler in SAMPLERS:
        if sampler.backend == backend:
            return sampler
    supported = ", ".join(sampler.backend for sampler in SAMPLERS)
    raise RuntimeError(
        f"no GPU memory profiler for backend {backend!r} (supported: {supported})"
    )


class GpuMemoryProfiler:
    def __init__(
        self,
        backend: Optional[str] = None,
        device: int = 0,
        interval_ms: int = DEFAULT_INTERVAL_MS,
    ) -> None:
        self.backend = backend or detect_backend()
        self.device = device
        self.interval_ms = interval_ms
        self._sampler_type = _sampler_type(self.backend)
        self._sampler: Optional[_MemorySampler] = None
        self._started_at = 0.0

    def start(self) -> None:
        if self._sampler is not None:
            raise RuntimeError("profiler is already running")
        sampler = self._sampler_type(self.device, self.interval_ms)
        sampler.start()
        self._sampler = sampler
        self._started_at = time.perf_counter()

    def stop(self) -> MemoryProfile:
        if self._sampler is None:
            raise RuntimeError("profiler was not started")
        duration = time.perf_counter() - self._started_at
        sampler, self._sampler = self._sampler, None
        samples = sampler.stop()
        return MemoryProfile(
            backend=self.backend,
            peak_mib=max(samples),
            mean_mib=sum(samples) / len(samples),
            baseline_mib=sampler.baseline,
            samples=len(samples),
            duration_s=duration,
        )
