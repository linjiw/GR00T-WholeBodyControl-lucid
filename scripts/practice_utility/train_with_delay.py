#!/usr/bin/env python3
"""Run SONIC training with delayed actuators enabled.

A thin launcher: it arranges for the actuator swap to happen at the one moment
it can, then hands off to ``gear_sonic/train_agent_trl.py`` with every argument
untouched. Keeping the swap here rather than in ``robots/g1.py`` means a
baseline arm run through the ordinary entrypoint is genuinely unmodified.

On the timing, which is fiddly and was wrong twice
--------------------------------------------------
``robots/g1.py`` cannot be imported until Isaac Sim exists, and Isaac Sim is
launched by ``AppLauncher`` *inside* ``train_agent_trl.main`` -- not at import
time. Patching straight after importing the module therefore failed with
``No module named 'isaaclab.envs'``.

Importing the module and calling ``main()`` directly failed differently: Hydra
resolves ``config_path="config"`` relative to the *module* that declares
``@hydra.main``, so as an import it looks for a package ``gear_sonic.config``,
which has no ``__init__.py``. Run as a script it resolves by file path and works.

So the file is run as a script, and the seam is moved somewhere that survives
that: ``custom_instantiate`` in ``gear_sonic.trl.utils.common``, which
``create_manager_env`` calls to build the environment config. Patching it in its
home module before the script runs means the script's ``from ... import
custom_instantiate`` picks up the patched version. By the time it is called for
the ``manager_env`` config, Isaac is up and the robot has not been resolved yet
-- exactly the window the swap needs.

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

    # Patch before the script imports it: `from ... import custom_instantiate`
    # copies the current binding, so this must be in place first.
    from gear_sonic.trl.utils import common as sonic_common

    original_instantiate = sonic_common.custom_instantiate
    state = {"patched": False}

    def instantiate_with_delay(d, *args, **kwargs):
        # Fires for every instantiate call; the actuator swap is only valid once
        # Isaac is up, which is true by the time any env config is built.
        if not state["patched"]:
            try:
                from gear_sonic.research.practice_utility.actuator_patch import (
                    describe_actuators,
                    enable_delayed_actuators,
                )

                report = enable_delayed_actuators(
                    max_delay=known.max_delay, min_delay=known.min_delay
                )
                state["patched"] = True
                print(
                    f"[latency] swapped {report['num_groups']} actuator groups, "
                    f"max_delay={known.max_delay} steps "
                    f"({known.max_delay * 5} ms at 200 Hz)",
                    flush=True,
                )
                for group, klass in describe_actuators().items():
                    print(f"[latency]   {group}: {klass}", flush=True)
                if report["num_groups"] == 0:
                    print("[latency] WARNING: nothing swapped; this run has NO latency",
                          flush=True)
            except ImportError:
                # Isaac not up yet; try again on the next instantiate call.
                pass
        return original_instantiate(d, *args, **kwargs)

    sonic_common.custom_instantiate = instantiate_with_delay

    entrypoint = REPO / "gear_sonic" / "train_agent_trl.py"
    sys.argv = [str(entrypoint), *passthrough]
    runpy.run_path(str(entrypoint), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
