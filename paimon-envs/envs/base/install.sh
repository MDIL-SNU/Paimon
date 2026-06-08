#!/bin/bash
# Base environment installation script
# Run once
# Usage: bash install.sh

set -e

ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAI_ROOT="$(cd "$ENV_DIR/../.." && pwd)"
source "$PAI_ROOT/setup_base.sh"

cd "$ENV_DIR"
if [ ! -f ".venv/bin/activate" ]; then
  uv venv --python 3.12.10
fi

source ".venv/bin/activate"
uv sync --active --inexact

######### Shared resources (in PAI_ROOT/external/) ########
mkdir -p "$PAI_ROOT/external"
cd "$PAI_ROOT/external"

# packmol (shared across all environments)
if [ ! -d "packmol" ]; then
  echo "Building packmol..."
  git clone https://github.com/m3g/packmol.git packmol
  cd packmol
  git checkout v21.0.1
  ./configure
  make
  cd ..
  echo "Packmol built."
fi

# pots directory for model potentials
mkdir -p "$PAI_ROOT/pots"

deactivate

echo "Base installation complete."
echo "Activate with: source $ENV_DIR/setup.sh"
