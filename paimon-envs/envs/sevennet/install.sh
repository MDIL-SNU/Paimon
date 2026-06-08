#!/bin/bash
# SevenNet environment installation script
# Run once to install flashTP and compile LAMMPS
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

######### SevenNet-specific installation ########
ENV_EXTERNAL="${ENV_DIR}/external"
mkdir -p "$ENV_EXTERNAL"
cd "$ENV_EXTERNAL"

  # flashTP (installed separately, uv does not track this)
  if [ ! -d "flashTP" ]; then
    git clone https://github.com/SNU-ARC/flashTP.git
  fi
  cd flashTP
    uv pip install -r requirements.txt
    CUDA_ARCH_LIST="80;86;89;90;120" uv pip install . --no-build-isolation
  cd ../
  
  # Set LD_LIBRARY_PATH for LAMMPS compilation
  export LD_LIBRARY_PATH="$ENV_DIR/.venv/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH"
  export LD_LIBRARY_PATH="$ENV_EXTERNAL/flashTP/build/lib.linux-x86_64-cpython-312/flashTP_e3nn:$LD_LIBRARY_PATH"
  export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"  # for CUDA-D3
  
  # LAMMPS with SevenNet patch
  if [ ! -d "lammps_sevenn" ]; then
    git clone https://github.com/lammps/lammps.git lammps_sevenn --branch stable_2Aug2023_update3 --depth=1
  fi
  sevenn patch_lammps ./lammps_sevenn --flashTP --d3
  cd ./lammps_sevenn
    mkdir -p build; cd build
      cmake -C ../cmake/presets/most.cmake ../cmake \
        -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
        -DWITH_GZIP=yes \
        -DUSE_SYSTEM_NVTX:BOOL=ON \
        -DPython3_ROOT_DIR=$HOME/.local/share/uv/python/cpython-3.12.10-linux-x86_64-gnu \
        -DPython3_INCLUDE_DIR=$HOME/.local/share/uv/python/cpython-3.12.10-linux-x86_64-gnu/include/python3.12 \
        -DPython3_LIBRARY=$HOME/.local/share/uv/python/cpython-3.12.10-linux-x86_64-gnu/lib/libpython3.12.so
      make -j24
  
      # link in uv venv bin dir
      ln -sf "$(realpath ./lmp)" "$ENV_DIR/.venv/bin/lmp"
    cd ../  # build/ out
  cd ../  # lammps_sevenn/ out
  
  # Potential deployment for lammps
  mkdir -p pots; cd pots
    [ -f SevenNet-0.pt ]    || sevenn get_model 7net-0    -flashTP -o SevenNet-0.pt
    [ -f SevenNet-l3i5.pt ] || sevenn get_model 7net-l3i5 -flashTP -o SevenNet-l3i5.pt
    [ -f SevenNet-omat.pt ] || sevenn get_model 7net-omat -flashTP -o SevenNet-omat.pt
    
    # Multi-fidelity: SEVENNET_{MODEL}_{MODALITY}
    [ -f SevenNet-mf-ompa_mpa.pt ]    || sevenn get_model 7net-mf-ompa -m mpa    -flashTP -o SevenNet-mf-ompa_mpa.pt
    [ -f SevenNet-mf-ompa_omat24.pt ] || sevenn get_model 7net-mf-ompa -m omat24 -flashTP -o SevenNet-mf-ompa_omat24.pt
    [ -f SevenNet-omni_mpa.pt ]       || sevenn get_model 7net-omni -m mpa       -flashTP -o SevenNet-omni_mpa.pt
  cd ../  # pots/ out

cd ../  # $ENV_EXTERNAL out

######### SevenNet-specific installation ########

deactivate

echo "SevenNet installation complete."
echo "Activate with: source $ENV_DIR/setup.sh"
