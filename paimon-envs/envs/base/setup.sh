#!/bin/bash
# Base environment activation script
# Usage: source setup.sh
# NOTE: Run install.sh first to compile env-specific binaries 

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

######### base-specific environment ########
ENV_EXTERNAL="${ENV_DIR}/external"

cd "$OLDPWD"
