the openMM script that has tol at 10.0

# INSIDE the BUILD script:
python3 relax_bonds_full6.py  -i frame_0000.pdb -o relaxed.pdb --krest_sc $2   --maxiter $3 --tol 10.0 --krest_bb $1

# The BUILD script:

time bash build_file.sh 5000 200   Mean RMSD: [1.18]

python3 relax_bonds_full5.py  -i frame_0000.pdb -o relaxed.pdb   --maxiter $2 --tol 7.0 --krest $1 

