import os
import cv2
import numpy as np
import h5py

character= {'0':0,
            '1':1,
            '2':2,
            '3':3,
            '4':4,
            '5':5,
            '6':6,
            '7':7,
            '8':8,
            '9':9,
            'ba':10,
            'pa':11}

pathTest = 'test/'
pathTrain = 'train/'

def prepareData(path):
    size = 0
    imageFiles = []
    labels = []
    for key in character.keys():
        files = os.listdir(path+key+'/')
        size += files.__len__()
        imageLabel = [0]*12
        imageLabel[character[key]] = 1
        for file in files:
            imageFiles.append(path+key+'/'+file)
            labels.append(imageLabel)

    data = np.zeros((size,50,50), dtype=np.double)
    for i in range(size):
        image = cv2.imread(imageFiles[i], cv2.IMREAD_GRAYSCALE)
        image = image/255
        data[i,:,:]=image
    dataLables = np.array(labels, dtype=np.double)
    return data, dataLables

def createHDF5(datas, labels, file):
    x = datas.astype(np.float32)
    y = labels.astype(np.float32)

    with h5py.File(file, 'w') as f:
        f.create_dataset('data', data=x, shape=x.shape)
        f.create_dataset('label', data=y, shape=y.shape)

def readHDF5(file):
    with h5py.File(file, 'r') as f:
        data = np.array(f.get('data'))
        label = np.array(f.get('label'))
        return data, label

if __name__=='__main__':
    data,label = prepareData(pathTrain)
    createHDF5(data, label, 'train.h5')
    data, label = prepareData(pathTest)
    createHDF5(data, label, 'test.h5')

