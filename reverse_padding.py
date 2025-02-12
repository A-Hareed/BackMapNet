import numpy as np
import sys


pred_arr = np.load(sys.argv[1])
mask = np.load(sys.argv[2])

valid_cols = np.any(mask, axis=0) 

arr_reduced = pred_arr[:, valid_cols]

np.save('predicted_arr_no_Pad.npy',arr_reduced)


