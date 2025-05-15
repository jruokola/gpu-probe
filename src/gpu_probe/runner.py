#!/usr/bin/env python3
"""
gpu_probe.runner — node-local GPU sanity tests
• runs run_nccl.sh  ➜  parses bandwidth
• runs run_gpu_burn.sh
Exits 0 if both pass.
"""

import logging
import os
import pathlib
import re
import subprocess
import sys

# ---------------------------------------------------------------------------
# config – resolved at runtime so it survives WORKDIR changes
# ---------------------------------------------------------------------------
CWD = pathlib.Path.cwd()
NCCL_SCRIPT = CWD / "run_nccl.sh"
GPU_BURN_SCRIPT = CWD / "run_gpu_burn.sh"
NCCL_OUTPUT = CWD / "nccl.txt"  # created by run_nccl.sh
SECONDS = 30  # default burn duration when wrapper omits arg

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)


def _run(cmd: str, tag: str) -> int:
    """Run *cmd* inside /bin/bash, stream stdout/stderr, return exit-code."""
    if not cmd:
        logging.error("%s: empty command", tag)
        return 1
    logging.info("%s: %s", tag, cmd)
    proc = subprocess.run(cmd, shell=True, text=True, executable="/bin/bash")
    logging.info("%s: exit %s", tag, proc.returncode)
    return proc.returncode


def _script_ok(path: pathlib.Path, tag: str) -> bool:
    if path.is_file() and os.access(path, os.X_OK):
        return True
    logging.error("%s missing or not executable: %s", tag, path)
    return False


def _parse_nccl(path: pathlib.Path):
    if not path.exists():
        logging.error("NCCL output not found: %s", path)
        return None
    txt = path.read_text()
    # match the *last* “8.0M … ” line
    for line in reversed(txt.splitlines()):
        if line.startswith("8.0M"):
            try:
                bw = float(line.split()[-2])  # GB/s value
                return bw
            except Exception:
                break
    # fallback to echo pattern “NCCL bw XXX GB/s”
    m = re.search(r"NCCL bw ([0-9.e+-]+) GB/s", txt)
    return float(m.group(1)) if m else None


def main():
    if "--test" not in sys.argv:
        logging.warning("runner.py is meant to be called with --test")
        sys.exit(0)

    node = os.getenv("SLURMD_NODENAME", "unknown")
    logging.info("Node-local GPU probe on %s", node)

    # -- NCCL test -----------------------------------------------------------
    if _script_ok(NCCL_SCRIPT, "NCCL"):  # presence check
        rc_nccl = _run(f"{NCCL_SCRIPT}", "NCCL")
    else:
        rc_nccl = 1

    bw = _parse_nccl(NCCL_OUTPUT) if rc_nccl == 0 else None
    if bw is not None:
        logging.info("NCCL bandwidth %.1f GB/s", bw)
    else:
        logging.error("Failed to parse NCCL bandwidth")

    # -- GPU-burn ------------------------------------------------------------
    if _script_ok(GPU_BURN_SCRIPT, "GPU-burn"):
        rc_burn = _run(f"{GPU_BURN_SCRIPT} {SECONDS}", "GPU-burn")
    else:
        rc_burn = 1

    # -- result --------------------------------------------------------------
    if rc_nccl == 0 and rc_burn == 0:  # SIGTERM and regular
        logging.info("✅  node-local GPU checks passed")
        sys.exit(0)
    logging.error(
        "❌  node-local GPU checks failed (NCCL=%s, burn=%s)", rc_nccl, rc_burn
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
