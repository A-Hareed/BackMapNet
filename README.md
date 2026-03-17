# BackMapNet
Deep-learning protein backmapping from coarse-grained (CG) beads to all-atom (AA) coordinates.

## 1. Project Overview
BackMapNet is a generalized deep-learning framework for reconstructing all-atom protein structures from coarse-grained inputs. Instead of relying on global fold-specific reconstruction rules, BackMapNet uses a local reconstruction strategy and decomposes the task into two coordinated models:
- Backbone reconstruction model
- Side-chain reconstruction model

This design improves transferability across proteins with different global structures and sequences. The framework was trained using 12 protein trajectories.

**Framework diagram (placeholder)**  
Add the final architecture/workflow figure for publication here.

```text
[Placeholder: BackMapNet framework diagram]
```

## 2. Installation And Dependencies
### 2.1 Requirements
- Bash (macOS/Linux)
- Python 3
- `numpy`
- `tensorflow` / `keras`
- `h5py` (recommended for model/weight compatibility workflows)

### 2.2 Recommended environment
```bash
conda create -n backmapnet python=3.11 -y
conda activate backmapnet
pip install numpy tensorflow h5py
```

### 2.3 Repository layout
- `BackMapNet.sh`: main user-facing entrypoint
- `bash_scripts/`: stage wrappers used by the main pipeline
- `python_scripts/`: data conversion, inference, reconstruction, and PDB generation
- `weights/`: backbone and side-chain model files

For normal usage, run the framework through `BackMapNet.sh` rather than individual internal scripts.

## 3. Atomic Mapping Specification
This section defines how BackMapNet maps CG representations to AA atoms.

### 3.1 Backbone mapping
- CG representation: one backbone bead (`BB`) per residue
- AA output per residue: `N, CA, C, O`
- Coordinate expansion: `3 -> 12` coordinates per residue
- Output order: residue-major, fixed atom order within each residue (`N, CA, C, O`)

### 3.2 Side-chain mapping
Side-chain reconstruction follows residue-specific heavy-atom templates. For each residue, the atom order is:

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

### 3.3 Local-frame side-chain representation
Side-chain reconstruction uses local-frame transforms during normalization/denormalization:
- `R_localFrame_<PDB>_cluster<ID>.npy`: local rotation terms
- `O_localFrame_<PDB>_cluster<ID>.npy`: local origins/translations
- `localFrame_META_<PDB>_cluster<ID>.npz`: frame metadata

### 3.4 Final merged coordinate layout
Final combined arrays (`combined_<PDB>_prediction.npy` or `combined_<PDB>_actual.npy`) are assembled per residue in chain order:
1. Backbone atoms: `N, CA, C, O`
2. Side-chain atoms: residue-specific ordering from Section 3.2

Chain segmentation is preserved from user-provided or auto-inferred chain lengths.

## 4. Running The Pipeline (`BackMapNet.sh`)
`BackMapNet.sh` is the public entrypoint for end-to-end execution.

### 4.1 Command help
```bash
bash /absolute/path/to/backbone/BackMapNet.sh --help
```

### 4.2 Common execution modes
#### CG-only mode (default)
```bash
bash /absolute/path/to/backbone/BackMapNet.sh \
  --pdb-name IgE \
  --cg-pdb-dir /data/IgE/cg \
  --jobs 8
```

#### Full mode (AA targets provided)
Providing `--aa-pdb-dir` switches to full mode automatically.
```bash
bash /absolute/path/to/backbone/BackMapNet.sh \
  --pdb-name IgE \
  --cg-pdb-dir /data/IgE/cg \
  --aa-pdb-dir /data/IgE/aa_backbone \
  --aa-sc-pdb-dir /data/IgE/aa_sidechain \
  --jobs 8
```

#### Export reconstructed PDB frames
```bash
bash /absolute/path/to/backbone/BackMapNet.sh \
  --pdb-name IgE \
  --cg-pdb-dir /data/IgE/cg \
  --write-pdb 1 \
  --pdb-frame-spec all
```

#### Debug run on a frame subset
```bash
bash /absolute/path/to/backbone/BackMapNet.sh \
  --pdb-name IgE \
  --cg-pdb-dir /data/IgE/cg \
  --frame-range 0-10 \
  --jobs 4
```

### 4.3 Typical outputs
- Backbone prediction arrays
- Side-chain prediction arrays
- Merged arrays: `combined_<PDB>_*`
- Optional PDB outputs: `pdb_frames_<PDB>/`
