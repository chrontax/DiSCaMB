"""Diagnostic helpers importable from the discamb container tests.

``GpuMemoryProfiler`` samples GPU memory between ``start()`` and ``stop()``
every ``DEFAULT_INTERVAL_MS`` milliseconds with the source of the backend it
detects: cuda -> ``nvidia-smi``; hip -> amdgpu sysfs (VRAM + GTT in use);
level_zero -> DRM fdinfo of the i915/xe clients in this container. ``start()`` takes a one-shot reading as
the baseline and fails there if the tool does not work; ``stop()`` returns a
``MemoryProfile`` holding the peak and the mean of the samples, in MiB.
"""

from __future__ import annotations

import glob
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


class _PollingSampler(_MemorySampler):
    """Reads a kernel interface from a thread instead of running a tool."""

    def read_once(self) -> float:
        raise NotImplementedError

    def start(self) -> None:
        self.baseline = self.read_once()
        self._stopping = threading.Event()
        self._reader = threading.Thread(target=self._poll, daemon=True)
        self._reader.start()

    def _poll(self) -> None:
        while True:
            try:
                self._samples.append(self.read_once())
            except (OSError, RuntimeError, ValueError) as error:
                if len(self._rejected) < 5:
                    self._rejected.append(str(error))
            if self._stopping.wait(self.interval_ms / 1000):
                return

    def stop(self) -> List[float]:
        self._stopping.set()
        self._reader.join(timeout=SAMPLER_STOP_TIMEOUT_S)
        if not self._samples:
            raise RuntimeError(
                f"{self.tool} produced no memory samples (errors: {self._rejected})"
            )
        return self._samples


MIB = 1024 * 1024


def _amdgpu_devices() -> List[str]:
    return sorted(
        os.path.dirname(path)
        for path in glob.glob("/sys/class/drm/card*/device/mem_info_vram_used")
    )


class AmdgpuSysfsSampler(_PollingSampler):
    backend = "hip"
    tool = "amdgpu sysfs"

    @classmethod
    def available(cls) -> bool:
        return bool(_amdgpu_devices())

    def read_once(self) -> float:
        devices = _amdgpu_devices()
        if self.device >= len(devices):
            raise RuntimeError(f"no amdgpu device {self.device} (found {devices})")
        used = 0
        # GTT too: integrated GPUs place most buffers in system memory.
        for name in ("mem_info_vram_used", "mem_info_gtt_used"):
            with open(os.path.join(devices[self.device], name)) as f:
                used += int(f.read())
        return used / MIB


_FDINFO_UNITS = {"": 1, "KiB": 1024, "MiB": MIB, "GiB": 1024 * MIB}


class IntelFdinfoSampler(_PollingSampler):
    """Sums drm-total-* of every i915/xe client open in this container.

    Neither driver exposes device-wide usage, and xpu-smi covers only data
    centre GPUs, so per-client DRM fdinfo is the one source that works for
    integrated and Arc GPUs alike. ``device`` is ignored.
    """

    backend = "level_zero"
    tool = "DRM fdinfo"

    @classmethod
    def available(cls) -> bool:
        for vendor in glob.glob("/sys/class/drm/renderD*/device/vendor"):
            with open(vendor) as f:
                if f.read().strip() == "0x8086":
                    return True
        return False

    def read_once(self) -> float:
        clients = {}
        for fd in glob.glob("/proc/[0-9]*/fd/*"):
            try:
                if not os.readlink(fd).startswith("/dev/dri/"):
                    continue
                with open(fd.replace("/fd/", "/fdinfo/", 1)) as f:
                    info = dict(
                        line.split(":", 1) for line in f.read().splitlines() if ":" in line
                    )
            except OSError:
                continue  # the process or descriptor went away meanwhile
            if info.get("drm-driver", "").strip() not in ("i915", "xe"):
                continue
            total = 0
            for key, value in info.items():
                if key.startswith("drm-total-"):
                    number, _, unit = value.strip().partition(" ")
                    total += int(number) * _FDINFO_UNITS[unit]
            client = (info.get("drm-pdev", "").strip(), info.get("drm-client-id", fd).strip())
            clients[client] = total
        return sum(clients.values()) / MIB


SAMPLERS: Tuple[Type[_MemorySampler], ...] = (
    NvidiaSmiSampler,
    AmdgpuSysfsSampler,
    IntelFdinfoSampler,
)


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
