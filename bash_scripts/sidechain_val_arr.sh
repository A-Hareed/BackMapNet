

for pdb in {0..399}
#for pdb in CG_frame_*.pdb
do
        python3 NEW_pdb2arr_CG.py CG_frame_"$pdb".pdb  cluster_2_CG.npy IgE
done




for pdb in {0..399}
#for pdb in frame_*.pdb
do
	sed 's/HID/HIS/g'  frame_"$pdb".pdb  > temp1.pdb
	sed 's/NGLNA/GLN A/g' temp1.pdb > temp2.pdb
	sed 's/NARGB/ARG B/g' temp2.pdb > temp3.pdb
	sed 's/CASPB/ASP B/g' temp3.pdb > temp4.pdb
	sed 's/CYX/CYS/g' temp4.pdb > temp5.pdb
	sed 's/NGLUA/GLU A/g' temp5.pdb > temp6.pdb
	sed 's/CLYSA/LYS A/g' temp6.pdb > temp7.pdb

	python3 NEW_pdb2arr_sideChain.py temp7.pdb  cluster_2_SC.npy
        rm -f temp*.pdb
done


for pdb in {0..399}
#for pdb in CG_frame_*.pdb
do
        python3 NEW_pdb2arr_CG_SC.py CG_frame_"$pdb".pdb  cluster_2_CG_SC.npy


done



exit
