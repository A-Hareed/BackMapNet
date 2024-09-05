import numpy as np
from sliding_window import create_feature_set
import sys


train_LAB = np.load(sys.argv[2])[:,:-4]
train_feat = np.load(sys.argv[1])[:,:-1]


test_LAB = np.load(sys.argv[4])[:,:-4]
test_feat = np.load(sys.argv[3])[:,:-1]

window_size = int(sys.argv[6])

print(train_feat.reshape(-1,3).shape, train_LAB.reshape(-1,12).shape,test_feat.shape,test_LAB.shape)


train_LAB_48 = create_feature_set(train_LAB,48,12)

test_LAB_48 = create_feature_set(test_LAB,48,12)

train_feat_12 = create_feature_set(train_feat,12,3)

test_feat_12 = create_feature_set(test_feat,12,3)




print(train_feat_12.shape, train_LAB_48.shape, test_feat_12.shape, test_LAB_48.shape)


print(int(sys.argv[5]))
if int(sys.argv[5]) == 1:
	np.save('final_train_feat.npy',train_feat_12)
	np.save('final_train_target.npy',train_LAB_48)

	np.save('final_test_feat.npy',test_feat_12)
	np.save('final_test_target.npy',test_LAB_48)
else:
	train_f = np.concatenate((np.load('final_train_feat.npy'),train_feat_12),axis=0)
	train_t = np.concatenate((np.load('final_train_target.npy'),train_LAB_48),axis=0)
	test_f = np.concatenate((np.load('final_test_feat.npy'),test_feat_12),axis=0)
	test_t = np.concatenate((np.load('final_test_target.npy'),test_LAB_48),axis=0)

	np.save('final_train_feat.npy',train_f)
	np.save('final_train_target.npy',train_t)

	np.save('final_test_feat.npy',test_f)
	np.save('final_test_target.npy',test_t)
