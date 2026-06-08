#!/bin/bash
# SevenNet environment activation script
# Usage: source setup.sh
# NOTE: Run install.sh first to install flashTP and compile LAMMPS

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

######### SevenNet-specific environment ########
ENV_EXTERNAL="${ENV_DIR}/external"

# PyTorch libraries (required for LAMMPS with libtorch)
export LD_LIBRARY_PATH="$ENV_DIR/.venv/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH"

# flashTP libraries
export LD_LIBRARY_PATH="$ENV_EXTERNAL/flashTP/build/lib.linux-x86_64-cpython-312/flashTP_e3nn:$LD_LIBRARY_PATH"

# Single models
export SEVENNET_0="${ENV_EXTERNAL}/pots/SevenNet-0.pt"
export SEVENNET_L3I5="${ENV_EXTERNAL}/pots/SevenNet-l3i5.pt"
export SEVENNET_OMAT="${ENV_EXTERNAL}/pots/SevenNet-omat.pt"

# Multi-fidelity: SEVENNET_{MODEL}_{MODALITY}
export SEVENNET_MFOMPA_MPA="${ENV_EXTERNAL}/pots/SevenNet-mf-ompa_mpa.pt"
export SEVENNET_MFOMPA_OMAT24="${ENV_EXTERNAL}/pots/SevenNet-mf-ompa_omat24.pt"
export SEVENNET_OMNI_MPA="${ENV_EXTERNAL}/pots/SevenNet-omni_mpa.pt"

cd "$OLDPWD"
