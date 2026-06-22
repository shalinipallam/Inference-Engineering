"""
Orchestrates the InferTutor mixed-track capstone experiment suite.

For each experiment it:
  1. Calls run_infertutor_experiment.py (deploy + health + load test).
  2. Immediately stops the resulting Modal app so GPUs do not idle-bill or stack.

Results JSON land in results_infertutor/. Failures are logged and the suite
continues with the next experiment.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
RUNNER = ROOT / "run_infertutor_experiment.py"

COMMON = ["--gpu-type", "H100", "--mode", "mixed", "--max-tokens", "96"]

# (label, [extra args]) -- defaults: max-seqs 32, max-batch-tokens 4096,
# prefix-cache on, chunked-prefill on, fast-boot (eager) on.
EXPERIMENTS = [
    # Baseline: 1 replica, default starter config.
    ("mm-baseline", ["--replicas", "1", "--users", "50", "--duration", "60", "--ramp-up", "15"]),
    # More users on 1 replica: find where p95 TTFT bends.
    ("mm-users100", ["--replicas", "1", "--users", "100", "--duration", "60", "--ramp-up", "15"]),
    # Prefix caching OFF (compare to baseline ON), same load.
    ("mm-noprefix", ["--replicas", "1", "--users", "50", "--duration", "60", "--ramp-up", "15", "--no-prefix-cache"]),
    # Chunked prefill OFF (compare to baseline ON), same load.
    ("mm-nochunked", ["--replicas", "1", "--users", "50", "--duration", "60", "--ramp-up", "15", "--no-chunked-prefill"]),
    # Wider batch knobs vs seq32/b4096.
    ("mm-wide", ["--replicas", "1", "--users", "50", "--duration", "60", "--ramp-up", "15", "--max-seqs", "64", "--max-batch-tokens", "8192"]),
    # Scale out to 2 replicas (matches internal reference 100u/2GPU).
    ("mm-scale2", ["--replicas", "2", "--users", "100", "--duration", "75", "--ramp-up", "20"]),
    # Scale out to 4 replicas (matches internal reference 120u/4GPU).
    ("mm-scale4", ["--replicas", "4", "--users", "120", "--duration", "75", "--ramp-up", "25"]),
]


def stop_app(label: str) -> None:
    app_name = f"infertutor-{label}".replace("_", "-")
    print(f"\n[suite] stopping app {app_name}", flush=True)
    try:
        subprocess.run(
            [sys.executable, "-m", "modal", "app", "stop", app_name, "--yes"],
            cwd=ROOT, timeout=180,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[suite] WARN: failed to stop {app_name}: {exc}", flush=True)


def run_one(label: str, extra: list[str]) -> bool:
    cmd = [sys.executable, str(RUNNER), "--label", label, *COMMON, *extra]
    print(f"\n{'=' * 70}\n[suite] EXPERIMENT {label}\n[suite] {' '.join(cmd)}\n{'=' * 70}", flush=True)
    ok = False
    try:
        proc = subprocess.run(cmd, cwd=ROOT, timeout=1800)
        ok = proc.returncode == 0
        print(f"[suite] {label} finished rc={proc.returncode}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[suite] ERROR: {label} raised {exc}", flush=True)
    finally:
        stop_app(label)
    return ok


def main() -> None:
    only = set(sys.argv[1:])  # optionally pass labels to run a subset
    results = {}
    t0 = time.time()
    for label, extra in EXPERIMENTS:
        if only and label not in only:
            continue
        results[label] = run_one(label, extra)
    dt = time.time() - t0
    print(f"\n{'#' * 70}\n[suite] DONE in {dt/60:.1f} min", flush=True)
    for label, ok in results.items():
        print(f"[suite]   {label}: {'OK' if ok else 'FAILED'}", flush=True)


if __name__ == "__main__":
    main()
