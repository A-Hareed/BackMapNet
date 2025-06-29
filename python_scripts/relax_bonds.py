#!/usr/bin/env python3
import sys, argparse
import numpy as np
from pdbfixer import PDBFixer
from itertools import combinations

import openmm
from openmm import unit
from openmm.app import (
    PDBFile, ForceField, Simulation,
    NoCutoff
)

def parse_args():
    p = argparse.ArgumentParser(
        description="Relax clashes & bond lengths, holding backbone in place"
    )
    p.add_argument("-i", "--input",  required=True,
                   help="Input PDB (heavy atoms only OK)")
    p.add_argument("-o", "--output", default="relaxed.pdb",
                   help="Output PDB with clashes relieved")
    p.add_argument("--krest", type=float, default=5000.0,
                   help="Backbone restraint k (kJ/mol/nm^2)")
    p.add_argument("--maxiter", type=int, default=150,
                   help="Max minimization iterations")
    p.add_argument("--tol",     type=float, default=7.0,
                   help="Minimization tolerance (kJ/mol/nm)")
    return p.parse_args()

def main():
    args = parse_args()

    # 1) PDBFixer
    print("🛠 Running PDBFixer...")
    fixer = PDBFixer(filename=args.input)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.0)

    # 1.5) Manual disulfide detection & CYX renaming
    print("🔗 Manually detecting disulfide bonds…")
    cys_res = [res for res in fixer.topology.residues() if res.name == 'CYS']
    pos_nm  = fixer.positions.value_in_unit(unit.nanometer)
    ss_pairs = []

    for r1, r2 in combinations(cys_res, 2):
        sg1 = next((a for a in r1.atoms() if a.name=='SG'), None)
        sg2 = next((a for a in r2.atoms() if a.name=='SG'), None)
        if sg1 and sg2:
            d = np.linalg.norm(pos_nm[sg1.index] - pos_nm[sg2.index])
            if d < 0.22:
                ss_pairs.append((r1, r2, sg1, sg2))

    for r1, r2, sg1, sg2 in ss_pairs:
        fixer.topology.addBond(sg1, sg2)
        r1.name = 'CYX'
        r2.name = 'CYX'
        print(f"  • Bonded SG {sg1.index} ↔ SG {sg2.index} and renamed {r1.id}, {r2.id} → CYX")

    topo = fixer.topology
    pos  = fixer.positions

    with open("fixed_input.pdb", "w") as f:
        PDBFile.writeFile(topo, pos, f)
    print("🛠 fixed_input.pdb written with S–S bonds and CYX labels")

    # 2) Build system (include external bonds)
    print("⚙️  Building OpenMM System...")
    ff = ForceField('amber/protein.ff14SB.xml')
    system = ff.createSystem(
        topo,
        nonbondedMethod=NoCutoff,
        constraints=None,
        ignoreExternalBonds=False
    )

    # 3) Restrain backbone heavy atoms (explicit squares)
    restraint = openmm.CustomExternalForce(
        "0.5*k*((x-x0)*(x-x0) + (y-y0)*(y-y0) + (z-z0)*(z-z0))"
    )
    for pname in ("x0","y0","z0","k"):
        restraint.addPerParticleParameter(pname)
    system.addForce(restraint)

    coords = pos.value_in_unit(unit.nanometer)
    nrest = 0
    for atom in topo.atoms():
        if atom.name in ("N","CA","C","CB"):
            idx = atom.index
            x0, y0, z0 = coords[idx]
            restraint.addParticle(idx, [x0, y0, z0, args.krest])
            nrest += 1
    print(f"🔗 Restrained {nrest} backbone atoms with k={args.krest} kJ/mol/nm^2")

    # 4) Simulation setup
    integrator = openmm.LangevinIntegrator(
        300*unit.kelvin,
        1.0/unit.picoseconds,
        0.001*unit.picoseconds
    )
    sim = Simulation(topo, system, integrator)
    sim.context.setPositions(pos)

    # 5) Minimize
    print(f"🔄 Minimizing (maxIter={args.maxiter}, tol={args.tol})…")
    sim.minimizeEnergy(
        tolerance=args.tol * unit.kilojoule_per_mole / unit.nanometer,
        maxIterations=args.maxiter
    )
    print("✅ Minimization complete")

    # 6) Write out relaxed PDB
    final_pos = sim.context.getState(getPositions=True).getPositions()
    with open(args.output, 'w') as out:
        PDBFile.writeFile(topo, final_pos, out)
    print(f"✅ Wrote relaxed PDB to: {args.output}")

if __name__ == "__main__":
    main()


# rm -f cluster_EM.npy 
# python3 NEW_pdb2arr_full.py my_minimized.pdb  cluster_EM.npy
# python3 relax_bonds_full3.py  -i pdb_frames/frame_0000.pdb -o relaxed.pdb


