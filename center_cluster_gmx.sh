

gmx trjconv -s topol.tpr -f traj.xtc -o protein_only.xtc -n index.ndx << EOF
1
EOF

gmx trjconv -s topol.tpr -f protein_only.xtc -o fitted_protein.xtc -fit rot+trans << EOF
1
EOF


echo "1 1" | gmx rms -s topol.tpr -f fitted_protein.xtc -m rmsd_protein.xpm -n index.ndx 


gmx cluster -s topol.tpr -f fitted_protein.xtc -dm rmsd_protein.xpm -o clusters_protein.pdb -g cluster_protein.log -cl cluster_centers_protein.pdb -method gromos -cutoff 0.2
