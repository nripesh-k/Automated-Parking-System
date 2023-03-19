import model as mdl
import prepareData as pd

if __name__ == "__main__":
    recognitionModel = mdl.model()
    recognitionModel.load_weights('trainedModel.h5')

    testData, testLabel = pd.readHDF5('test.h5')
    testScore = recognitionModel.evaluate(testData, testLabel, verbose=0)
    print(f'Test Loss: {testScore[0]}')
    print(f'Test Accuracy: {testScore[1]*100}%')
