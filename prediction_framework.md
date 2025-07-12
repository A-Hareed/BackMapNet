# normalise the CG values 
python3 new_sideChain_Normalisation.py $CG_CLUSTER_SIDE_CHAIN $PDB $CG_CLUSTER_BB_FILE

# optional evalutation step for AA normalisation:
python3 get_scaled_side_chain_model_AA.py $AA_CLUSTER_FILE $PDB $CG_CLUSTER_FILE

# to get the predicted values use this:
python3 new_train_test_minmax.py cluster_1_SC_CG_RBF_minMax.npy cluster_PD_1_SC_MinMax.npy masking_input_1.npy 1

# reverse scale the prediction and remove the padding

python3 reverse_nonPAD.py yhat_frame_1_IgE.npy sequence_IgE.txt custom_range_IgE_perBead1.npy custom_min_IgE_perBead_1.npy masking_input_1.npy IgE pred cluster_PD_1_SC_MinMax.npy




# RELAXING BOND ANGLES:
time bash build_file.sh pdb_frames/frame_0.pdb 1000 1000 0


{build_file.sh >>>} 
python3 relax_bonds_full11.py -i $1 -o relaxed.pdb   --maxiter $3 --tol 0.1 --krest $2
