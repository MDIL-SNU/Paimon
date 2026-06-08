#!/bin/bash
# Validates sevennet environment integrity after setup.sh is sourced.
# Usage: source setup.sh && bash check.sh
set -e

# SevenNet
python - <<'PY'
import importlib.util
import sys

packages = ["sevenn", "flashTP_e3nn"]
missing = [p for p in packages if importlib.util.find_spec(p) is None]
if missing:
    print("Missing packages:", ", ".join(missing))
    sys.exit(1)
sys.exit(0)
PY

# LAMMPS binary
lmp -h > /dev/null 2>&1

# At least one model file must exist
test -f "$SEVENNET_0"
