#!/bin/bash
# Validates mace environment integrity after setup.sh is sourced.
# Usage: source setup.sh && bash check.sh
set -e

# MACE
# SevenNet
python - <<'PY'
import importlib.util
import sys

packages = ["mace"]
missing = [p for p in packages if importlib.util.find_spec(p) is None]
if missing:
    print("Missing packages:", ", ".join(missing))
    sys.exit(1)
sys.exit(0)
PY

# LAMMPS binary (_lmp is wrapped by .venv/bin/lmp)
_lmp -h > /dev/null 2>&1

# At least one model file must exist
test -f "$MACE_MPA_0_MEDIUM"
