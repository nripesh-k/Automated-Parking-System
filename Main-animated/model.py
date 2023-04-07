import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.optimizers import Adam

character = {
    0:'0',
    1:'1',
    2:'2',
    3:'3',
    4:'4',
    5:'5',
    6:'6',
    7:'7',
    8:'8',
    9:'9',
    10:'Ba',
    11:'Pa',
}

NLPD_characters = {
    0:'0',
    1:'1',
    2:'2',
    3:'3',
    4:'4',
    5:'5',
    6:'6',
    7:'7',
    8:'8',
    9:'9',
    10:'B',
    11:'C',
    12:'P',
}


def NLPD_model():
    model = Sequential()
    model.add(Conv2D(filters=32,kernel_size=(5,5), activation='relu', padding='valid',\
                    use_bias=True, input_shape=(32,32,1)))
    model.add(MaxPool2D(pool_size=(2,2), strides=2))

    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', padding='valid'))
    model.add(MaxPool2D(pool_size=(2,2), strides=2))
    
    model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='valid'))
    # model.add(MaxPool2D(pool_size=(2,2), strides=2))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dense(96, activation = 'relu'))
    model.add(Dense(13, activation='softmax'))

    adam = Adam(learning_rate=0.001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def model():
    model = Sequential()
    model.add(Conv2D(filters=32,kernel_size=(5,5), activation='relu', padding='valid',\
                    use_bias=True, input_shape=(32,32,1)))
    model.add(MaxPool2D(pool_size=(2,2), strides=2))

    model.add(Conv2D(filters=64, kernel_size=(5,5), activation='relu', padding='valid'))
    model.add(MaxPool2D(pool_size=(2,2), strides=2))
    
    model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='valid'))
    model.add(MaxPool2D(pool_size=(2,2), strides=2))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(12, activation='softmax'))

    adam = Adam(learning_rate=0.001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

if __name__ == '__main__':
    recognitionModel = model()
    recognitionModel.load_weights('weights.h5')
    files = os.listdir('numberPlates/2/')
    number = files.__len__()
    numberPlate = ''
    for i in range(1,number+1):
        image = cv2.imread(f'numberPlates/2/2_{i}.jpg', cv2.IMREAD_GRAYSCALE)
        image = image/255
        img = np.zeros((1,50,50),dtype=np.float32)
        img[0,:,:] = image
        prediction = recognitionModel.predict(img)
        numberPlate+=character[np.argmax(prediction[0])]
    
    print(numberPlate)