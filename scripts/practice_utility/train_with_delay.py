#!/usr/bin/env python3
"""Run SONIC training with delayed actuators enabled.

A thin launcher: it enables the delayed actuators, prints what it swapped, then
hands off to ``gear_sonic/train_agent_trl.py`` with every argument untouched.
Keeping the swap here rather than in ``robots/g1.py`` means a baseline arm run
through the ordinary entrypoint is genuinely unmodified.

    python scripts/practice_utility/train_with_delay.py --max-delay 8 -- \
        +exp=manager/universal_token/all_modes/sonic_release ...
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-delay", type=int, default=8,
                        help="delay-buffer capacity in physics steps "
                             "(8 = 40 ms at 200 Hz, LUCID's training range)")
    parser.add_argument("--min-delay", type=int, default=0)
    known, passthrough = parser.parse_known_args(argv)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    # Isaac Sim must be launched before isaaclab imports resolve, and
    # train_agent_trl does that at import time via AppLauncher. The swap has to
    # land after that import but before the env config is built, so the module
    # is imported first and patched immediately afterwards.
    import gear_sonic.train_agent_trl  # noqa: F401  (launches the app)

    from gear_sonic.research.practice_utility.actuator_patch import (
        describe_actuators,
        enable_delayed_actuators,
    )

    report = enable_delayed_actuators(max_delay=known.max_delay, min_delay=known.min_delay)
    print(f"[latency] swapped {report['num_groups']} actuator groups, "
          f"max_delay={known.max_delay} steps "
          f"({known.max_delay * 5} ms at 200 Hz)")
    for group, klass in describe_actuators().items():
        print(f"[latency]   {group}: {klass}")
    if report["num_groups"] == 0:
        print("[latency] WARNING: nothing was swapped; this run has NO latency")

    sys.argv = [str(REPO / "gear_sonic" / "train_agent_trl.py"), *passthrough]
    runpy.run_path(str(REPO / "gear_sonic" / "train_agent_trl.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
