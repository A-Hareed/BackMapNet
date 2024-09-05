import numpy as np
from sliding_window import create_feature_set
import sys



test_LAB = np.load(sys.argv[2])[:,:-4]
test_feat = np.load(sys.argv[1])[:,:-1]

window_size = int(sys.argv[4])

print(test_feat.shape,test_LAB.shape)


lab_window = window_size*12
feat_window = window_size*3

test_LAB_48 = create_feature_set(test_LAB,lab_window,12)


test_feat_12 = create_feature_set(test_feat,feat_window,3)




print( test_feat_12.shape, test_LAB_48.shape)


print(int(sys.argv[3]))
if int(sys.argv[3]) == 1:


	np.save(f'final_test_feat_{window_size}.npy',test_feat_12)
	np.save(f'final_test_target_{window_size}.npy',test_LAB_48)
else:
	
	test_f = np.concatenate((np.load(f'final_test_feat_{window_size}.npy'),test_feat_12),axis=0)
	test_t = np.concatenate((np.load(f'final_test_target_{window_size}.npy'),test_LAB_48),axis=0)



	np.save(f'final_test_feat_{window_size}.npy',test_f)
	np.save(f'final_test_target_{window_size}.npy',test_t)
