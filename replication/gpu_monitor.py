"""Read-only NVML sampling, including a wrapper for arbitrary GPU test commands.

Run with the repository venv: python -m replication.gpu_monitor --out <json>
-- <command> [args]. Whole-process energy includes loading/compilation; use
summarize_samples with explicit workload timestamps for benchmark comparisons.
"""

import argparse
import bisect
import json
from pathlib import Path
import statistics
import subprocess
import threading
import time


def summarize_samples(samples, start, end, new_tokens=None):
    """Integrate reported board watts over a fully bracketed time interval."""
    if end <= start:
        raise ValueError("measurement interval must have positive duration")
    if new_tokens is not None and new_tokens <= 0:
        raise ValueError("new_tokens must be positive")
    points = sorted((s["monotonic"], s["power_w"]) for s in samples
                    if s.get("power_w") is not None)
    if len(points) < 2 or points[0][0] > start or points[-1][0] < end:
        raise ValueError("power samples must bracket the measurement interval")
    times = [t for t, _ in points]

    def at(t):
        i = max(0, min(len(points) - 2, bisect.bisect_right(times, t) - 1))
        a, b = points[i], points[i + 1]
        return a[1] + (b[1] - a[1]) * (t - a[0]) / (b[0] - a[0])

    clipped = [(start, at(start))] + [(t, p) for t, p in points if start < t < end] + [(end, at(end))]
    joules = sum((b[0] - a[0]) * (a[1] + b[1]) / 2
                 for a, b in zip(clipped, clipped[1:]))
    inside = [s for s in samples if start <= s["monotonic"] <= end]
    summary = {"start_monotonic": start, "end_monotonic": end,
               "duration_seconds": end - start, "sample_count_inside": len(inside),
               "average_board_power_w": joules / (end - start),
               "peak_sampled_board_power_w": max(p for _, p in clipped),
               "estimated_board_energy_j": joules,
               "max_sample_gap_seconds": max(b[0] - a[0] for a, b in zip(points, points[1:])
                                             if a[0] < end and b[0] > start)}
    if new_tokens is not None:
        summary.update(new_tokens=new_tokens, new_tokens_per_second=new_tokens / (end - start),
                       estimated_board_joules_per_new_token=joules / new_tokens)
    for field in ("gpu_busy_percent", "memory_busy_percent", "sm_clock_mhz", "memory_clock_mhz",
                  "temperature_c", "memory_used_mib"):
        values = [s[field] for s in inside if s.get(field) is not None]
        if values:
            summary[field] = {"median": statistics.median(values), "min": min(values), "max": max(values)}
    summary["clock_event_reason_masks"] = sorted({s["clock_event_reasons"] for s in inside
                                                   if s.get("clock_event_reasons") is not None})
    return summary


class PowerMonitor:
    """Sample board telemetry without changing clocks, power limits or drivers."""

    def __init__(self, interval=0.1, device=0):
        if interval <= 0:
            raise ValueError("interval must be positive")
        self.interval, self.device = interval, device
        self.samples, self.errors = [], []
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def __enter__(self):
        import pynvml as nv
        self.nv = nv
        nv.nvmlInit()
        try:
            self.handle = (nv.nvmlDeviceGetHandleByUUID(self.device) if isinstance(self.device, str)
                           else nv.nvmlDeviceGetHandleByIndex(self.device))
            self.metadata = {"gpu": nv.nvmlDeviceGetName(self.handle),
                             "uuid": nv.nvmlDeviceGetUUID(self.handle),
                             "driver": nv.nvmlSystemGetDriverVersion(),
                             "power_limit_w": nv.nvmlDeviceGetEnforcedPowerLimit(self.handle) / 1000,
                             "interval_seconds": self.interval,
                             "clock_event_mask_definitions": {
                                 name: getattr(nv, name) for name in dir(nv)
                                 if name.startswith("nvmlClocksEventReason") and isinstance(getattr(nv, name), int)},
                             "limits": ["NVML-reported board power may be internally averaged; faster polling does not improve sensor resolution.",
                                        "Peak is a sampled peak, not a guaranteed instantaneous maximum.",
                                        "Energy is integrated board power, including desktop/other GPU activity; not whole-system energy.",
                                        "Short workload windows have limited power/energy accuracy."]}
            self.sample()
            if self.samples[0]["power_w"] is None:
                raise RuntimeError("NVML power monitoring is unavailable")
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            return self
        except BaseException:
            nv.nvmlShutdown()
            raise

    def sample(self):
        nv, h = self.nv, self.handle

        def read(fn, *args):
            try:
                return fn(h, *args)
            except nv.NVMLError:
                return None

        with self._lock:
            power = read(nv.nvmlDeviceGetPowerUsage)
            util = read(nv.nvmlDeviceGetUtilizationRates)
            memory = read(nv.nvmlDeviceGetMemoryInfo)
            reasons = getattr(nv, "nvmlDeviceGetCurrentClocksEventReasons",
                              nv.nvmlDeviceGetCurrentClocksThrottleReasons)
            self.samples.append({"monotonic": time.perf_counter(), "unix_time": time.time(),
                                 "power_w": power / 1000 if power is not None else None,
                                 "gpu_busy_percent": util.gpu if util else None,
                                 "memory_busy_percent": util.memory if util else None,
                                 "sm_clock_mhz": read(nv.nvmlDeviceGetClockInfo, nv.NVML_CLOCK_SM),
                                 "memory_clock_mhz": read(nv.nvmlDeviceGetClockInfo, nv.NVML_CLOCK_MEM),
                                 "temperature_c": read(nv.nvmlDeviceGetTemperature, nv.NVML_TEMPERATURE_GPU),
                                 "memory_used_mib": memory.used / 2**20 if memory else None,
                                 "clock_event_reasons": read(reasons)})

    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                self.sample()
            except Exception as exc:
                self.errors.append(repr(exc))
                break

    def __exit__(self, *exc):
        self._stop.set()
        self.thread.join()
        try:
            self.sample()
        finally:
            self.nv.nvmlShutdown()

    def save(self, path, **extra):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"metadata": self.metadata, "errors": self.errors,
                                    "samples": self.samples, **extra}, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--interval", type=float, default=0.1)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        ap.error("a command is required after --")
    code = 1
    with PowerMonitor(args.interval) as monitor:
        start = time.perf_counter()
        try:
            code = subprocess.call(command)
        finally:
            end = time.perf_counter()
    summary = summarize_samples(monitor.samples, start, end)
    monitor.save(args.out, command=command, returncode=code, whole_process=summary)
    print(json.dumps({"telemetry": str(args.out), "returncode": code,
                      "average_board_power_w": summary["average_board_power_w"],
                      "peak_sampled_board_power_w": summary["peak_sampled_board_power_w"]}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
