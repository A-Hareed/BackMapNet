import numpy as np
import csv
import sys
import os



# PDB = 'CG_bb.pdb'
PDB = sys.argv[1]
array_file = sys.argv[2]
name_pdb = sys.argv[3]

    
with open(PDB, "r") as f:
    row = []
    full_data = []
    sequence = []
    segment_starts = [0]
    bb_res_idx = -1
    prev_res_uid = None
    prev_chain = None
    prev_resseq_int = None
    ter_pending = False

    for raw in f:
        rec = raw[0:6].strip()

        if rec == "TER":
            if row:
                full_data.append(row)
                row = []
            prev_res_uid = None
            ter_pending = True
            continue

        if rec != "ATOM":
            continue

        atom_name = raw[12:16].strip()   # e.g. CH3, CB, OXT
        res_name  = raw[17:20].strip()   # e.g. ACE, MET
        chain_id  = raw[21:22]           # chain label
        res_seq   = raw[22:26].strip()   # residue index
        ins_code  = raw[26:27]
        x = raw[30:38].strip()
        y = raw[38:46].strip()
        z = raw[46:54].strip()

        # Skip ACE and OXT
        if res_name == "ACE" or atom_name == "OXT":
            continue

        # New residue detection uses chain+resseq+icode.
        res_uid = (chain_id, res_seq, ins_code)
        if res_uid != prev_res_uid:
            bb_res_idx += 1
            sequence.append(res_name)

            try:
                resseq_int = int(res_seq)
            except ValueError:
                resseq_int = None

            # Segment break if:
            # 1) explicit TER before this residue
            # 2) chain label changed
            # 3) residue index reset/decreased (e.g. 100 -> 1)
            new_segment = False
            if bb_res_idx > 0:
                if ter_pending:
                    new_segment = True
                elif prev_chain is not None and chain_id != prev_chain:
                    new_segment = True
                elif (
                    prev_resseq_int is not None
                    and resseq_int is not None
                    and resseq_int < prev_resseq_int
                ):
                    new_segment = True

            if new_segment:
                segment_starts.append(bb_res_idx)

            prev_chain = chain_id
            prev_resseq_int = resseq_int
            prev_res_uid = res_uid
            ter_pending = False

        row.extend([x, y, z])

    if row:
        full_data.append(row)





    
#print(full_data[:3])
    # print(row)



# Check if the file exists
file_name = f'sequence_{name_pdb}.txt'

# If the file doesn't exist, open and write to it
if not os.path.exists(file_name):
    with open(file_name, 'w') as f:
        result = ''
        for i, res in enumerate(sequence):
            if i < (len(sequence) - 1):
                result += res + ','
            else:
                result += res
        f.write(result)
else:
    print(f"File {file_name} already exists. Skipping.")

# Save detected segment starts for local_frames.py argv[5].
seg_file = f'segment_starts_{name_pdb}.csv'
segment_starts = sorted(set(segment_starts))
if not os.path.exists(seg_file):
    with open(seg_file, 'w') as f:
        f.write(",".join(str(i) for i in segment_starts))
    print(f"Saved segment starts to {seg_file}: {segment_starts}")
else:
    print(f"File {seg_file} already exists. Skipping.")



#with open(f'sequence_{name_pdb}.txt','w') as f:
#    result=''
#    for i,res in enumerate(sequence):
#        if i < (len(sequence)-1):
#            result += res +','
#        else:
#            result += res
#    f.write(result)


arr = np.array(full_data)


if not os.path.exists(array_file):
    np.save(array_file,arr)
    
    print(f"Coordinates saved to {array_file}")
else:
    print(f"{array_file} already exists. Skipping file.")
    
    old_arr = np.load(array_file)
    arr = np.concatenate((old_arr,arr),axis=0)

    np.save(array_file,arr)
