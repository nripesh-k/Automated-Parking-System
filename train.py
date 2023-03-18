from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPool2D, Flatten
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import numpy as np
import prepareData as pd

def model():
    model = Sequential()
    model.add(Conv2D(filters=32,kernel_size=(5,5), activation='relu', padding='valid',\
                    use_bias=True, input_shape=(50,50,1)))
    model.add(MaxPool2D(pool_size=(2,2), strides=2))

    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', padding='valid'))
    model.add(MaxPool2D(pool_size=(2,2), strides=2))
    
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='valid'))
    model.add(MaxPool2D(pool_size=(2,2), strides=2))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(12, activation='softmax'))

    adam = Adam(learning_rate=0.001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def train():
    modelCNN = model()
    print(modelCNN.summary())

    data, label = pd.readHDF5('train.h5')
    tData, tLabel = pd.readHDF5('test.h5')

    checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='min')
    callbacks_list = [checkpoint]

    modelCNN.fit(data, label, batch_size=20, validation_data = (tData,tLabel), callbacks=callbacks_list, shuffle=True, epochs=300, verbose = 0)
    modelCNN.save('trainedModel.h5')

if __name__=="__main__":
    train()