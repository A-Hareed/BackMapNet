import tensorflow as tf
import numpy as np
import sys

frame_num = int(sys.argv[2])
pdb = sys.argv[1]
print('current frame number: ',frame_num)

#boxsize = np.array([22.50000,  22.50000,  22.50000]) *10


feat_lst = [f'train_feat_B1_{pdb}_chain{frame_num}.npy']

lab_lst = [f'train_LAB_B1_{pdb}_chain{frame_num}.npy']

model = tf.keras.models.load_model(sys.argv[3])

print(model.summary())

for i in range(400):
    #if i==0:
    feat_arr = np.load(f'train_feat_B{i}_{pdb}_chain{frame_num}.npy')
    lab_arr = np.load(f'train_LAB_B{i}_{pdb}_chain{frame_num}.npy')

    model.evaluate(feat_arr,lab_arr)
    yhat = model.predict(feat_arr)
    np.save(f'yhat_frame_{i}_chain_{frame_num}.npy',yhat)

exit()
#
#    else:
#        feat_arr = np.concatenate((feat_arr,np.load(feat_lst[i])),axis=0)
#        lab_arr = np.concatenate((lab_arr, np.load(lab_lst[i])),axis =0)
#    print(feat_arr.shape)

#"""
#feat_arr = np.load(feat_lst[0])
#lab_arr = np.load(lab_lst[0])



#model = tf.keras.models.load_model('model5_check_epoch_03.keras')
#model = tf.keras.models.load_model(sys.argv[3])

# Assume validation data is loaded in variables X_val and y_val
# X_val: features, y_val: true labels




# Make predictions on the validation data
history = model.evaluate(feat_arr,lab_arr)

yhat = model.predict(feat_arr)

np.save(f'yhat_whole_{frame_num}.npy',yhat)

print(history)


exit()

