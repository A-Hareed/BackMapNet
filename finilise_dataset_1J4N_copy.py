import numpy as np
from sliding_window import create_feature_set
import sys
import pickle



test_LAB = np.load(sys.argv[2])[:,:-1]
test_feat = np.load(sys.argv[1])[:,:-1]

with open(sys.argv[5],'rb') as fp:
	slices = pickle.load(fp)

slice_feat = slices[0]
slice_target =  slices[1]

if sys.argv[5] == "1J4N_chain_index":
	test_Lab_a1 = test_LAB[:,:slice_target[0]]
	test_Lab_a2 = test_LAB[:,slice_target[0]:slice_target[1]]
	test_Lab_a3 = test_LAB[:,slice_target[1]:slice_target[2]]
	test_Lab_a4 = test_LAB[:,slice_target[2]:]
	#--------------------------------------------
	test_feat_a1 = test_feat[:,:slice_feat[0]]
	test_feat_a2 = test_feat[:,slice_feat[0]:slice_feat[1]]
	test_feat_a3 = test_feat[:,slice_feat[1]:slice_feat[2]]
	test_feat_a4 = test_feat[:,slice_feat[2]:]

	window_size = int(sys.argv[4])


	print(test_feat.shape,test_LAB.shape)


	lst_test_lab = []
	lst_test_feat = []
	lab_window = window_size*12
	feat_window = window_size*3


	test_LAB_48_a1 = create_feature_set(test_Lab_a1,lab_window,12)
	test_LAB_48_a2 = create_feature_set(test_Lab_a2,lab_window,12)
	test_LAB_48_a3 = create_feature_set(test_Lab_a3,lab_window,12)
	test_LAB_48_a4 = create_feature_set(test_Lab_a4,lab_window,12)

	test_feat_12_a1 = create_feature_set(test_feat_a1,feat_window,3)
	test_feat_12_a2 = create_feature_set(test_feat_a2,feat_window,3)
	test_feat_12_a3 = create_feature_set(test_feat_a3,feat_window,3)
	test_feat_12_a4 = create_feature_set(test_feat_a4,feat_window,3)


	lst_test_lab.append(test_LAB_48_a1)
	lst_test_lab.append(test_LAB_48_a2)
	lst_test_lab.append(test_LAB_48_a3)
	lst_test_lab.append(test_LAB_48_a4)

	lst_test_feat.append(test_feat_12_a1)
	lst_test_feat.append(test_feat_12_a2)
	lst_test_feat.append(test_feat_12_a3)
	lst_test_feat.append(test_feat_12_a4)



else:

	test_Lab_a1 = test_LAB[:,:slice_target[0]]
	test_Lab_a2 = test_LAB[:,slice_target[0]:slice_target[1]]
	test_Lab_a3 = test_LAB[:,slice_target[1]:]
	
	#--------------------------------------------
	test_feat_a1 = test_feat[:,:slice_feat[0]]
	test_feat_a2 = test_feat[:,slice_feat[0]:slice_feat[1]]
	test_feat_a3 = test_feat[:,slice_feat[1]:]

	window_size = int(sys.argv[4])
	print(test_feat.shape,test_LAB.shape)


	lst_test_lab = []
	lst_test_feat = []
	lab_window = window_size*12
	feat_window = window_size*3



	test_LAB_48_a1 = create_feature_set(test_Lab_a1,lab_window,12)
	test_LAB_48_a2 = create_feature_set(test_Lab_a2,lab_window,12)
	test_LAB_48_a3 = create_feature_set(test_Lab_a3,lab_window,12)

	test_feat_12_a1 = create_feature_set(test_feat_a1,feat_window,3)
	test_feat_12_a2 = create_feature_set(test_feat_a2,feat_window,3)
	test_feat_12_a3 = create_feature_set(test_feat_a3,feat_window,3)


	lst_test_lab.append(test_LAB_48_a1)
	lst_test_lab.append(test_LAB_48_a2)
	lst_test_lab.append(test_LAB_48_a3)

	lst_test_feat.append(test_feat_12_a1)
	lst_test_feat.append(test_feat_12_a2)
	lst_test_feat.append(test_feat_12_a3)



test_LAB_48 = np.concatenate(lst_test_lab, axis=0)
test_feat_12 = np.concatenate(lst_test_feat, axis =0)

print(test_feat_12.shape, test_LAB_48.shape)

print(int(sys.argv[3]))
if int(sys.argv[3]) == 1:


	np.save(f'final_test_feat_{window_size}.npy',test_feat_12)
	np.save(f'final_test_target_{window_size}.npy',test_LAB_48)
else:

	test_f = np.concatenate((np.load(f'final_test_feat_{window_size}.npy'),test_feat_12),axis=0)
	test_t = np.concatenate((np.load(f'final_test_target_{window_size}.npy'),test_LAB_48),axis=0)


	np.save(f'final_test_feat_{window_size}.npy',test_f)
	np.save(f'final_test_target_{window_size}.npy',test_t)
