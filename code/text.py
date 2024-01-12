import os
import readData
import scipy.io as sio
import numpy as np
import tensorflow as tf2
import tensorflow.compat.v1 as tf
import keras.backend as K
from keras import optimizers
from tqdm import tqdm
from keras.models import load_model

os.environ['CUDA_VISIBLE_DEVICES']='0'
Model_Path = "..." 
Save_Path = "..." 
Test_Path = "..."

def jaccard_loss(y_true, y_pred):

    y_true = tf.squeeze(y_true)
    y_pred = tf.squeeze(y_pred)

    intersection = tf2.reduce_sum(tf2.multiply(y_true, y_pred))
    union = tf2.reduce_sum(y_true) + tf2.reduce_sum(y_pred) - intersection
    loss = 1. - (intersection / (union + K.epsilon()))

    return loss

filesdir = os.listdir(Test_Path)
adam = optimizers.Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model = load_model(Model_Path + 'model.h5', compile = False)
model.compile(optimizer = adam, loss = jaccard_loss)

for i in tqdm(range(len(filesdir))):
    x_test, y_test = readData.get_total_test(Test_Path,filesdir[i])
    intermediate_output = model.predict(x_test / np.max(x_test), batch_size = 1)   
    print("Data saving...")
    sio.savemat(Save_Path + filesdir[i],{'outputs':np.array(intermediate_output),'targets':y_test})    



