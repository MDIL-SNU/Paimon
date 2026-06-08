# Paimon Multi-Environment Setup

## Philosophy

Paimon agents require different ML interatomic potentials (MLIPs), each with conflicting dependencies. This setup provides isolated virtual environments where:

1. **Dependencies are declarative**: All packages in `pyproject.toml`, not hidden in shell scripts
2. **Builds are reproducible**: `uv.lock` pins exact versions
3. **Environments are isolated**: Each MLIP has its own `.venv` with custom LAMMPS binaries
4. **Utilities are shared**: `simulation_util` package is editable across all environments

## Environments

- **base**: Common tools (ASE, pymatgen, MDAnalysis, etc.)
- **sevennet**: Base + PyTorch + SevenNet + patched LAMMPS
- **mace**: Base + PyTorch + MACE (TODO: + patched LAMMPS)

## Structure

```
paimon-envs/
├── envs/{base,sevennet,mace}/
│   ├── pyproject.toml          # Dependencies (declarative)
│   ├── uv.lock                 # Version lock file
│   ├── .venv/                  # Virtual environment
│   ├── external/               # Environment-specific binaries (LAMMPS)
│   ├── install.sh              # One-time setup (compile binaries)
│   └── setup.sh                # Activation (source per session)
├── sub/simulation_util/        # Shared utilities (editable install)
├── external/packmol/           # Shared binaries
├── pots/                       # Pretrained models (.pt files) # TODO: move into envs
└── setup_base.sh               # HPC module loading (shared)
```

**Key Design**:
- Each environment has isolated `.venv` and `external/` binaries
- LAMMPS compiled per-environment (depends on PyTorch from venv)
- `simulation_util` shared as editable dependency

## Installation

Each environment requires two steps:

```bash
cd envs/{base,sevennet,mace}
bash install.sh      # One-time: compile binaries, setup .venv
source setup.sh      # Per-session: activate environment
```

**install.sh**: Creates `.venv`, syncs packages, compiles binaries (LAMMPS, packmol)
**setup.sh**: Loads HPC modules, syncs packages, activates venv

## Usage

```bash
# Activate environment
source envs/{base,sevennet,mace}/setup.sh

# Use Python packages
python -c "from simulation_util.lammps import parse_log"

# Use LAMMPS (sevennet/mace only)
lmp -in input.lammps
```

## Adding Dependencies

```bash
# 1. Edit envs/{env}/pyproject.toml
# 2. Sync environment
cd envs/{env} && uv sync

# For simulation_util changes:
# Edit sub/pyproject.toml, then sync all environments
cd envs/base && uv sync
cd envs/sevennet && uv sync
cd envs/mace && uv sync
```

## Adding New MLIP Environment

1. Create `envs/new_mlip/pyproject.toml` (copy from sevennet/mace as template)
2. Create `envs/new_mlip/setup.sh` (activation script - copy and modify)
3. Create `envs/new_mlip/install.sh` (binary compilation script)
4. Run `bash install.sh && source setup.sh`

See existing environments for reference templates.
