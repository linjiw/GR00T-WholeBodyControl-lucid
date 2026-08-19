#!/usr/bin/env python3
"""Run SONIC training with delayed actuators enabled.

A thin launcher: it arranges for the actuator swap to happen at the one moment
it can, then hands off to ``gear_sonic/train_agent_trl.py`` with every argument
untouched. Keeping the swap here rather than in ``robots/g1.py`` means a
baseline arm run through the ordinary entrypoint is genuinely unmodified.

On the timing, which is fiddly and was wrong once
-------------------------------------------------
``robots/g1.py`` cannot be imported until Isaac Sim exists, and Isaac Sim is
launched by ``AppLauncher`` *inside* ``train_agent_trl.main`` -- not at import
time, as an earlier version of this script assumed. Patching right after
importing the module therefore failed with ``No module named 'isaaclab.envs'``.

The swap must land after the app launches but before the environment config
resolves the robot, and there is exactly one seam that satisfies both:
``create_manager_env``, which calls ``custom_instantiate(config.manager_env)``
and is itself called well after the launcher. So that function is wrapped.

``main`` is invoked directly rather than through ``runpy``: running the file
again would build a fresh module object and quietly discard the patch.

    python scripts/practice_utility/train_with_delay.py --max-delay 8 -- \
        +exp=manager/universal_token/all_modes/sonic_release ...
"""

from __future__ import annotations

import argparse
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

    import gear_sonic.train_agent_trl as trainer

    original_create_env = trainer.create_manager_env

    def create_manager_env_with_delay(config, device, args_cli):
        # Isaac Sim is up by now, so robots/g1.py and the delayed actuator are
        # importable; the env config has not yet resolved the robot.
        from gear_sonic.research.practice_utility.actuator_patch import (
            describe_actuators,
            enable_delayed_actuators,
        )

        report = enable_delayed_actuators(
            max_delay=known.max_delay, min_delay=known.min_delay
        )
        print(
            f"[latency] swapped {report['num_groups']} actuator groups, "
            f"max_delay={known.max_delay} steps ({known.max_delay * 5} ms at 200 Hz)",
            flush=True,
        )
        for group, klass in describe_actuators().items():
            print(f"[latency]   {group}: {klass}", flush=True)
        if report["num_groups"] == 0:
            print("[latency] WARNING: nothing was swapped; this run has NO latency", flush=True)
        return original_create_env(config, device, args_cli)

    trainer.create_manager_env = create_manager_env_with_delay

    sys.argv = [str(REPO / "gear_sonic" / "train_agent_trl.py"), *passthrough]
    trainer.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
