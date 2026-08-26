#!/usr/bin/env python3
"""Run SONIC evaluation with delayed actuators enabled.

The runtime seam is the same one used by ``train_with_delay.py``; only the
upstream entrypoint changes to ``gear_sonic/eval_agent_trl.py``.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-delay",
        type=int,
        default=12,
        help="delay-buffer capacity in 5 ms physics steps",
    )
    parser.add_argument("--min-delay", type=int, default=0)
    known, passthrough = parser.parse_known_args(argv)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    from gear_sonic.trl.utils import common as sonic_common

    original_instantiate = sonic_common.custom_instantiate
    state = {"template_patched": False, "resolved_patched": False}

    def instantiate_with_delay(d, *args, **kwargs):
        if not state["template_patched"]:
            try:
                from gear_sonic.research.practice_utility.actuator_patch import (
                    describe_actuators,
                    enable_delayed_actuators,
                )

                report = enable_delayed_actuators(
                    max_delay=known.max_delay, min_delay=known.min_delay
                )
                state["template_patched"] = True
                print(
                    f"[latency] swapped {report['num_groups']} actuator groups, "
                    f"max_delay={known.max_delay} steps ({known.max_delay * 5} ms at 200 Hz)",
                    flush=True,
                )
                for group, klass in describe_actuators().items():
                    print(f"[latency]   {group}: {klass}", flush=True)
                if report["num_groups"] == 0:
                    print("[latency] WARNING: nothing swapped; this run has NO latency", flush=True)
            except ImportError:
                pass
        result = original_instantiate(d, *args, **kwargs)

        if not state["resolved_patched"]:
            robot_cfg = getattr(getattr(result, "scene", None), "robot", None)
            if robot_cfg is not None:
                from gear_sonic.research.practice_utility.actuator_patch import (
                    enable_delayed_actuators_on_cfg,
                )

                resolved = enable_delayed_actuators_on_cfg(
                    robot_cfg, max_delay=known.max_delay, min_delay=known.min_delay
                )
                state["resolved_patched"] = True
                print(
                    f"[latency] resolved env cfg has {resolved['num_groups']} delayed "
                    f"actuator groups: {resolved['groups']}",
                    flush=True,
                )
        return result

    sonic_common.custom_instantiate = instantiate_with_delay

    entrypoint = REPO / "gear_sonic" / "eval_agent_trl.py"
    sys.argv = [str(entrypoint), *passthrough]
    runpy.run_path(str(entrypoint), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
