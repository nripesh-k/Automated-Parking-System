from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import prepareData as pd
import model as mdl

def train():
    modelCNN = mdl.model()
    print(modelCNN.summary())

    data, label = pd.readHDF5('train.h5')
    tData, tLabel = pd.readHDF5('test.h5')

    checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='min')
    callbacks_list = [checkpoint]

    trainingHistory = modelCNN.fit(data, label, batch_size=15, validation_data = (tData,tLabel), callbacks=callbacks_list, shuffle=True, epochs=50, verbose = 0)
    modelCNN.save('trainedModel.h5')

    # summarize history for accuracy
    plt.plot(trainingHistory.history['accuracy'])
    plt.plot(trainingHistory.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(trainingHistory.history['loss'])
    plt.plot(trainingHistory.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

if __name__=="__main__":
    train()