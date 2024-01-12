import os
import numpy as np
import scipy.io as sio
from tqdm import tqdm
import random
import h5py

def normalization(inputs):

    return inputs / np.max(inputs) 

def gettest(files_path, files):

    filelist = list()
    targetlist = list()
  
    inputs = np.zeros([128,128,128,3])
        
    try:
        inputs[:,:,:,0] = normalization(np.squeeze(np.expand_dims(np.array(sio.loadmat(files_path + files)["f0"], dtype = np.float32), -1)))
        inputs[:,:,:,1] = normalization(np.squeeze(np.expand_dims(np.array(sio.loadmat(files_path + files)["f1"], dtype = np.float32), -1)))
        inputs[:,:,:,2] = normalization(np.squeeze(np.expand_dims(np.array(sio.loadmat(files_path + files)["f2"], dtype = np.float32), -1)))
        
        target = np.sign(np.expand_dims(np.array(sio.loadmat(files_path + files)["tumor"], dtype = np.float32), -1))
    except:
        inputs[:,:,:,0] = np.transpose(normalization(np.squeeze(np.expand_dims(np.array(h5py.File(files_path + files)["f0"][:], dtype = np.float32), -1))))
        inputs[:,:,:,1] = np.transpose(normalization(np.squeeze(np.expand_dims(np.array(h5py.File(files_path + files)["f1"][:], dtype = np.float32), -1))))
        inputs[:,:,:,2] = np.transpose(normalization(np.squeeze(np.expand_dims(np.array(h5py.File(files_path + files)["f2"][:], dtype = np.float32), -1))))
        
        target = np.transpose(np.sign(np.expand_dims(np.array(h5py.File(files_path + files)["tumor"][:], dtype = np.float32), -1)))

    filelist.append(inputs)
    targetlist.append(target)

    return np.array(filelist), np.array(targetlist)
