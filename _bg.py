"""Detached background job runner (Windows-friendly).

Usage:
  python _bg.py <logfile> <module_or_script> [args...]

Redirects stdout/stderr at the *Python* level (no inherited OS handles),
so the launching shell returns immediately.
"""

import os
import runpy
import sys


def main() -> None:
    logfile, script = sys.argv[1], sys.argv[2]
    rest = sys.argv[3:]
    log_dir = os.path.dirname(os.path.abspath(logfile))
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    sys.stdout = open(logfile + ".log", "w", encoding="utf-8", buffering=1)
    sys.stderr = open(logfile + ".err", "w", encoding="utf-8", buffering=1)
    sys.argv = [script] + rest
    sys.dont_write_bytecode = True
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
