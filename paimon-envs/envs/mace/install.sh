#!/bin/bash
# Mace environment installation script
# Run once to compile LAMMPS
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

######### mace-specific installation ########
ENV_EXTERNAL="${ENV_DIR}/external"
mkdir -p "$ENV_EXTERNAL"
cd "$ENV_EXTERNAL"

  # LAMMPS
  # Pin AMPERE86 as the lowest GPU architecture
  # This lmp is runnable also on a5000 and blackwell pro 6000.
  # most.cmake could not be applied due to using 'nvcc_wrapper'
  if [ ! -d "lammps_mace" ]; then
    git clone https://github.com/lammps/lammps.git lammps_mace
    cd lammps_mace
    git checkout ccca772
    cd ../
  fi
  cd lammps_mace
    mkdir -p build; cd build
      cmake -C ../cmake/presets/kokkos-cuda.cmake ../cmake \
        -D CMAKE_BUILD_TYPE=Release \
        -D CMAKE_INSTALL_PREFIX=$(pwd) \
        -D PKG_ML-IAP=ON \
        -D PKG_ML-SNAP=ON \
        -D MLIAP_ENABLE_PYTHON=ON \
        -D PKG_PYTHON=ON \
        -D BUILD_SHARED_LIBS=ON \
        -D Kokkos_ARCH_AMPERE86=ON \
        -DWITH_GZIP=yes -DPKG_EXTRA-DUMP=yes -DPKG_EXTRA-FIX=yes -DPKG_EXTRA-COMPUTE=yes -DPKG_MANYBODY=yes -DPKG_MC=yes \
        -DUSE_SYSTEM_NVTX:BOOL=ON \
        -DPython3_ROOT_DIR=$HOME/.local/share/uv/python/cpython-3.12.10-linux-x86_64-gnu \
        -DPython3_INCLUDE_DIR=$HOME/.local/share/uv/python/cpython-3.12.10-linux-x86_64-gnu/include/python3.12 \
        -DPython3_LIBRARY=$HOME/.local/share/uv/python/cpython-3.12.10-linux-x86_64-gnu/lib/libpython3.12.so
  
      make -j36
      python -m ensurepip --upgrade  # required since uv python do not have its pip
      make install-python
  
      # link lmp wrapper in uv venv bin dir
      ln -sf "$(realpath ./lmp)" "$ENV_DIR/.venv/bin/_lmp"
      cat > "$ENV_DIR/.venv/bin/lmp" <<'EOF'
#!/bin/bash
exec _lmp -k on g 1 -sf kk -pk kokkos newton on neigh half "$@"
EOF
      chmod +x "$ENV_DIR/.venv/bin/lmp"

    cd ../  # build/ out
  cd ../  # lammps_mace out

  # Potential deployment for lammps
  mkdir -p pots; cd pots
    DEPLOY_SCRIPT="${VIRTUAL_ENV}/lib/python3.12/site-packages/mace/cli/create_lammps_model.py"
    
    wget -nc https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model
    MACE_MODEL="mace-mpa-0-medium.model"
    if [ ! -f "${MACE_MODEL}-mliap_lammps.pt" ]; then
      python ${DEPLOY_SCRIPT} ${MACE_MODEL} --format=mliap
    fi
    
    wget -nc https://github.com/ACEsuit/mace-mp/releases/download/mace_omat_0/mace-omat-0-medium.model
    MACE_MODEL="mace-omat-0-medium.model"
    if [ ! -f "${MACE_MODEL}-mliap_lammps.pt" ]; then
      python ${DEPLOY_SCRIPT} ${MACE_MODEL} --format=mliap
    fi
  cd ../  # pots/ out

cd ../  # $ENV_EXTERNAL out

######### mace-specific installation ########

deactivate

echo "Mace installation complete."
echo "Activate with: source $ENV_DIR/setup.sh"
