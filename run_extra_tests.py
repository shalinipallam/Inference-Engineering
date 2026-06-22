"""Run two additional mixed-track experiments and stop each app afterward.

1. mm-wide-seqs : seq64 / b4096 (1 GPU, 50u) -- isolates max-seqs from the wide win.
2. mm-scale4-200: 4 replicas at 200 users -- 4-GPU scale-out at ~50 users/replica.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
RUNNER = ROOT / "run_infertutor_experiment.py"
COMMON = ["--gpu-type", "H100", "--mode", "mixed", "--max-tokens", "96"]

EXPERIMENTS = [
    ("mm-wide-seqs", ["--replicas", "1", "--users", "50", "--duration", "60",
                       "--ramp-up", "15", "--max-seqs", "64", "--max-batch-tokens", "4096"]),
    ("mm-scale4-200", ["--replicas", "4", "--users", "200", "--duration", "75",
                        "--ramp-up", "35"]),
]


def stop_app(label: str) -> None:
    app_name = f"infertutor-{label}".replace("_", "-")
    print(f"\n[extra] stopping app {app_name}", flush=True)
    try:
        subprocess.run([sys.executable, "-m", "modal", "app", "stop", app_name, "--yes"],
                       cwd=ROOT, timeout=180)
    except Exception as exc:  # noqa: BLE001
        print(f"[extra] WARN: stop {app_name} failed: {exc}", flush=True)


def main() -> None:
    for label, extra in EXPERIMENTS:
        cmd = [sys.executable, str(RUNNER), "--label", label, *COMMON, *extra]
        print(f"\n{'=' * 70}\n[extra] {label}\n[extra] {' '.join(cmd)}\n{'=' * 70}", flush=True)
        try:
            rc = subprocess.run(cmd, cwd=ROOT, timeout=1800).returncode
            print(f"[extra] {label} rc={rc}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[extra] ERROR {label}: {exc}", flush=True)
        finally:
            stop_app(label)
    print("\n[extra] DONE", flush=True)


if __name__ == "__main__":
    main()
