#!/bin/bash
# Base HPC module loading (shared across all environments)

module purge &> /dev/null
module add compiler/2022.1.0 &> /dev/null
module add mkl/2022.1.0 &> /dev/null
module add mpi/2021.6.0 &> /dev/null
module add VASP/basic_tools &> /dev/null
module add CUDA/12.8.1 &> /dev/null

export PAI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="${PAI_ROOT}/external/packmol:$PATH"
export PYTHONPATH="$PAI_ROOT/sitecustomize:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
