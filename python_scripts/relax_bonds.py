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
        description="Relax clashes & bond lengths, holding backbone or all atoms in place"
    )
    p.add_argument("-i", "--input",  required=True,
                   help="Input PDB (heavy atoms only OK)")
    p.add_argument("-o", "--output", default="relaxed.pdb",
                   help="Output PDB with clashes relieved")
    p.add_argument("--krest", type=float, default=1000.0,
                   help="Default restraint k (kJ/mol/nm^2)")
    p.add_argument("--maxiter", type=int, default=5000,
                   help="Max minimization iterations")
    p.add_argument("--tol",     type=float, default=1.0,
                   help="Minimization tolerance (kJ/mol/nm)")
    p.add_argument(
        "--types", default="",
        help="Comma-separated list of residue types (e.g. ALA,GLY,LYS) to restrain extra-tightly"
    )
    p.add_argument(
        "--krest_type", type=float, default=None,
        help="If set, use this k for atoms in residues of the specified types"
    )
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
    print("🔗 Detecting disulfide bonds…")
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

    PDBFile.writeFile(topo, pos, open("fixed_input.pdb", "w"))
    print("🛠 fixed_input.pdb written with S–S bonds and CYX labels")

    # 2) Build system
    print("⚙️ Building OpenMM System...")
    ff = ForceField('amber/protein.ff14SB.xml')
    system = ff.createSystem(
        topo,
        nonbondedMethod=NoCutoff,
        constraints=None,
        ignoreExternalBonds=False
    )

    # 3) Restrain atoms & optional residue-type specific restraints
    restraint = openmm.CustomExternalForce(
        "0.5*k*((x-x0)*(x-x0) + (y-y0)*(y-y0) + (z-z0)*(z-z0))"
    )
    for pname in ("x0","y0","z0","k"):
        restraint.addPerParticleParameter(pname)
    system.addForce(restraint)

    coords = pos.value_in_unit(unit.nanometer)
    trusted_types = set([t.strip().upper() for t in args.types.split(",") if t.strip()])
    total_restrained = 0

    for atom in topo.atoms():
        # Restrain every atom in the model
        idx = atom.index
        x0, y0, z0 = coords[idx]

        # Determine force constant
        resname = atom.residue.name.upper()
        if resname in trusted_types and args.krest_type is not None:
            k_here = args.krest_type
        else:
            k_here = args.krest

        restraint.addParticle(idx, [x0, y0, z0, k_here])
        total_restrained += 1

    print(f"🔗 Restrained {total_restrained} atoms (with type-specific k where set)")

    # 4) Setup simulation
    integrator = openmm.LangevinIntegrator(
        300*unit.kelvin,
        1.0/unit.picoseconds,
        0.001*unit.picoseconds
    )
    sim = Simulation(topo, system, integrator)
    sim.context.setPositions(pos)

    # 5) Minimize energy
    print(f"🔄 Minimizing (maxIter={args.maxiter}, tol={args.tol})…")
    sim.minimizeEnergy(
        tolerance=args.tol * unit.kilojoule_per_mole / unit.nanometer,
        maxIterations=args.maxiter
    )
    print("✅ Minimization complete")

    # 6) Write output
    final_pos = sim.context.getState(getPositions=True).getPositions()
    with open(args.output, 'w') as out:
        PDBFile.writeFile(topo, final_pos, out)
    print(f"✅ Wrote relaxed PDB to: {args.output}")

if __name__ == "__main__":
    main()
