#!/bin/bash
# Mace environment activation script
# Usage: source setup.sh
# NOTE: Run install.sh first to compile LAMMPS

OLDPWD=$(pwd)

ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAI_ROOT="$(cd "$ENV_DIR/../.." && pwd)"
source "$PAI_ROOT/setup_base.sh"

cd "$ENV_DIR"
if [ ! -f ".venv/bin/activate" ]; then
  echo "Error: venv not found."
  echo "Please run install.sh first."
  return 1 2>/dev/null || exit 1
fi

source ".venv/bin/activate"
uv sync --active --inexact 2>/dev/null

######### mace-specific environment ########
ENV_EXTERNAL="${ENV_DIR}/external"

# Python libraries for lammps 
export LD_LIBRARY_PATH="$(
python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))'
):$LD_LIBRARY_PATH"

# Potentials
export MACE_MPA_0_MEDIUM="${ENV_EXTERNAL}/pots/mace-mpa-0-medium.model-mliap_lammps.pt"
export MACE_OMAT_0_MEDIUM="${ENV_EXTERNAL}/pots/mace-omat-0-medium.model-mliap_lammps.pt"

######### mace-specific environment ########

cd "$OLDPWD"
