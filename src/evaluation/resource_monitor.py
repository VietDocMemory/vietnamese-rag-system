"""Low-overhead host and NVIDIA GPU sampling for RAG benchmarks on Linux."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path


def _read_meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip().split()[0]) * 1024
    except (OSError, ValueError):
        pass
    return values


def _read_cpu_totals() -> tuple[int, int] | None:
    try:
        fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()
        counters = [int(value) for value in fields[1:]]
    except (OSError, ValueError, IndexError):
        return None
    idle = counters[3] + (counters[4] if len(counters) > 4 else 0)
    return sum(counters), idle


def _gpu_rows() -> list[dict]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return []
    fields = [
        "index",
        "name",
        "driver_version",
        "memory.total",
        "memory.used",
        "utilization.gpu",
        "power.draw",
        "power.limit",
    ]
    try:
        result = subprocess.run(
            [
                executable,
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []

    rows = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != len(fields):
            continue

        def number(value: str) -> float | None:
            try:
                return float(value)
            except ValueError:
                return None

        rows.append(
            {
                "index": int(values[0]),
                "name": values[1],
                "driver_version": values[2],
                "memory_total_mib": number(values[3]),
                "memory_used_mib": number(values[4]),
                "utilization_percent": number(values[5]),
                "power_draw_w": number(values[6]),
                "power_limit_w": number(values[7]),
            }
        )
    return rows


def machine_inventory(path: str | Path = ".") -> dict:
    meminfo = _read_meminfo()
    disk = shutil.disk_usage(Path(path).resolve())
    cpu_model = platform.processor()
    if Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "ram_total_gib": meminfo.get("MemTotal", 0) / 2**30,
        "disk_total_gib": disk.total / 2**30,
        "disk_free_gib": disk.free / 2**30,
        "gpus": _gpu_rows(),
    }


@dataclass
class ResourceSample:
    monotonic_seconds: float
    system_cpu_percent: float | None
    ram_used_gib: float | None
    ram_available_gib: float | None
    gpu_memory_used_mib: float | None
    gpu_utilization_percent: float | None
    gpu_power_draw_w: float | None


class ResourceMonitor:
    """Continuously sample whole-machine resources used by the complete RAG stack."""

    def __init__(self, interval_seconds: float = 0.5):
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.interval_seconds = interval_seconds
        self.samples: list[ResourceSample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._previous_cpu = _read_cpu_totals()

    def _sample(self) -> ResourceSample:
        current_cpu = _read_cpu_totals()
        cpu_percent = None
        if current_cpu and self._previous_cpu:
            total_delta = current_cpu[0] - self._previous_cpu[0]
            idle_delta = current_cpu[1] - self._previous_cpu[1]
            if total_delta > 0:
                cpu_percent = 100 * (total_delta - idle_delta) / total_delta
        self._previous_cpu = current_cpu

        memory = _read_meminfo()
        total = memory.get("MemTotal")
        available = memory.get("MemAvailable")
        used = total - available if total is not None and available is not None else None
        gpus = _gpu_rows()
        return ResourceSample(
            monotonic_seconds=time.perf_counter(),
            system_cpu_percent=cpu_percent,
            ram_used_gib=used / 2**30 if used is not None else None,
            ram_available_gib=available / 2**30 if available is not None else None,
            gpu_memory_used_mib=sum(
                row["memory_used_mib"] or 0 for row in gpus
            )
            if gpus
            else None,
            gpu_utilization_percent=max(
                (row["utilization_percent"] or 0 for row in gpus), default=None
            ),
            gpu_power_draw_w=sum(row["power_draw_w"] or 0 for row in gpus) if gpus else None,
        )

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.samples.append(self._sample())

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("resource monitor is already running")
        self._stop.clear()
        self.samples.append(self._sample())
        self._thread = threading.Thread(target=self._run, name="rag-resource-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=max(5, self.interval_seconds * 2))
        self._thread = None

    def mark(self) -> int:
        return len(self.samples)

    def rows_since(self, index: int) -> list[dict]:
        return [asdict(sample) for sample in self.samples[index:]]

    def __enter__(self) -> "ResourceMonitor":
        self.start()
        return self

    def __exit__(self, *_args) -> None:
        self.stop()
