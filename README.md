# BackMapNet
BackMapNet is a deep-learning framework for reconstructing all-atom protein coordinates from coarse-grained (CG) trajectories.

A preprint describing BackMapNet is available on ChemRxiv: **“Generalised Protein BackMapper Using Machine Learning Models”** — [ChemRxiv preprint](https://doi.org/10.26434/chemrxiv.15004975/v1).

## Overview
BackMapNet performs local coordinate reconstruction with two coordinated models:

- A backbone model that predicts `N, CA, C, O` per residue.
- A side-chain model that predicts residue-specific heavy-atom coordinates in local frames.

This split improves transferability across proteins with different global folds and sequences.  
The models were trained on 12 protein trajectories.


![Alt text](Framework_labelled.png) 



## Tested Software Matrix
The repository does not currently include a lockfile; the matrix below reflects the active environment used for this project on March 17, 2026.

| Profile | Conda env | Python | NumPy | TensorFlow | Keras | h5py | Intended use |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Runtime | `mytfenv311` | `3.11.15` | `2.4.2` | `2.20.0` | `3.13.1` | `3.16.0` | BackMapNet pipeline execution |

To print your exact runtime versions:

```bash
python3 - <<'PY'
import importlib, sys
print("python", sys.version.split()[0])
for name in ["numpy", "tensorflow", "keras", "h5py"]:
    try:
        mod = importlib.import_module(name)
        print(name, mod.__version__)
    except Exception:
        print(name, "MISSING")
PY
```

## Installation
Create and activate an environment, then install core dependencies:

```bash
conda create -n backmapnet python=3.11 -y
conda activate backmapnet
pip install numpy tensorflow keras h5py
```

## Repository Layout
Top-level structure:

- `BackMapNet.sh`: public pipeline entrypoint.
- `run_all.sh`: backward-compatible wrapper that forwards to `BackMapNet.sh`.
- `bash_scripts/`: stage-level shell workflows.
- `python_scripts/`: array builders, model evaluation, reconstruction, and PDB writing.
- `weights/`: backbone/side-chain model files and priors.

## Input Conventions
BackMapNet supports either frame-indexed CG directories or one single CG PDB file:

- CG directory (`--cg-pdb-dir`): contains frame-indexed files named `CG_frame_<idx>.pdb`
- Single CG PDB (`--cg-pdb-file`): accepts any existing PDB filename, for example `12as_cg.pdb`
- Backbone AA directory (`--aa-pdb-dir`, optional): `frame_<idx>.pdb`
- Side-chain AA directory (`--aa-sc-pdb-dir`, required when full side-chain mode is used): `frame_<idx>_SC.pdb`

When `--cg-pdb-file` is used, BackMapNet internally stages that file as `CG_frame_0.pdb` in a temporary directory, so the rest of the pipeline still uses the same frame-indexed logic. In this mode, `--frame-range` must be `auto` or `0`.

If `--aa-pdb-dir` is provided, BackMapNet automatically switches from CG-only mode to full mode.

## Running BackMapNet
Show CLI help:

```bash
bash /absolute/path/to/backbone/BackMapNet.sh --help
```

### CG-only mode (default)
For a directory of frame-indexed CG PDBs:

```bash
bash /absolute/path/to/backbone/BackMapNet.sh \
  --pdb-name IgE \
  --cg-pdb-dir /data/IgE/cg \
  --jobs 8
```

For one CG PDB file with any filename:

```bash
bash /absolute/path/to/backbone/BackMapNet.sh \
  --pdb-name IgE \
  --cg-pdb-file /data/pdb_CG/IgE_cg.pdb
```

The single-file CG-only command does not require you to rename the input to `CG_frame_0.pdb`; BackMapNet creates that temporary staged name automatically. PDB export is enabled by default, so `--pdb-output-dir` only changes where the final PDB is written.

### Full mode (backbone + side-chain targets) Evaluation 
```bash
bash /absolute/path/to/backbone/BackMapNet.sh \
  --pdb-name IgE \
  --cg-pdb-dir /data/IgE/cg \
  --aa-pdb-dir /data/IgE/aa_backbone \
  --aa-sc-pdb-dir /data/IgE/aa_sidechain \
  --jobs 8
```

### PDB export
PDB export is enabled by default. If `--pdb-output-dir` is omitted, files are written to `pdb_frames_<PDB>`.

```bash
bash /absolute/path/to/backbone/BackMapNet.sh \
  --pdb-name IgE \
  --cg-pdb-dir /data/IgE/cg \
  --pdb-frame-spec all
```

To skip PDB writing and only keep NumPy arrays, pass `--write-pdb 0`.

## Output Files
Typical outputs are generated in the run directory:

- `backbone_<PDB>_prediction.npy`
- `backbone_<PDB>_actual.npy` (full mode only)
- `sidechain_<PDB>_prediction.npy`
- `combined_<PDB>_prediction.npy` (CG-only)
- `combined_<PDB>_actual.npy` (full mode)
- `pdb_frames_<PDB>/` (default PDB output directory)

## Atomic Mapping Specification
This section defines the atom ordering used by reconstruction and PDB writing.

### Backbone mapping
Per residue:

| CG feature | Reconstructed atoms | Atom order |
| --- | --- | --- |
| `BB` bead | Backbone heavy atoms | `N, CA, C, O` |

### Side-chain mapping
Per residue, side-chain heavy atoms are produced in this fixed order:

| Residue | Side-chain atom order |
| --- | --- |
| ALA | `CB` |
| ARG | `CB, CG, CD, NE, CZ, NH1, NH2` |
| ASN | `CB, CG, OD1, ND2` |
| ASP | `CB, CG, OD1, OD2` |
| CYS | `CB, SG` |
| GLN | `CB, CG, CD, OE1, NE2` |
| GLU | `CB, CG, CD, OE1, OE2` |
| GLY | *(no side-chain atoms)* |
| HIS | `CB, CG, ND1, CE1, NE2, CD2` |
| ILE | `CB, CG2, CG1, CD` |
| LEU | `CB, CG, CD1, CD2` |
| LYS | `CB, CG, CD, CE, NZ` |
| MET | `CB, CG, SD, CE` |
| PHE | `CB, CG, CD1, CE1, CZ, CE2, CD2` |
| PRO | `CD, CG, CB` |
| SER | `CB, OG` |
| THR | `CB, CG2, OG1` |
| TRP | `CB, CG, CD1, NE1, CE2, CZ2, CH2, CZ3, CE3, CD2` |
| TYR | `CB, CG, CD1, CE1, CZ, OH, CE2, CD2` |
| VAL | `CB, CG1, CG2` |

### Final merged array order
In `combined_<PDB>_*.npy`, each residue is assembled as:

1. Backbone atoms: `N, CA, C, O`
2. Side-chain atoms: residue-specific order from the table above


## License
This project is licensed under the MIT License.
You may use, copy, modify, merge, publish, distribute, sublicense, and/or sell this software, provided the copyright notice and license text are included in copies or substantial portions.

