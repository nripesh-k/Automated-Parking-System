import cv2
import numpy as np
import model as mdl

if __name__=='__main__':
    
    numberPlate = ''
    recognitoinModel = mdl.model()
    recognitoinModel.load_weights('weights.h5')
    
    id = 2
    image = cv2.imread(f"numberPlates/{id}.jpg", cv2.IMREAD_GRAYSCALE)
    
    maskedImage = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,-2)

    connections = cv2.connectedComponentsWithStats(maskedImage, 2, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = connections

    for i in range(1,numLabels):
        y = stats[i, cv2.CC_STAT_TOP]
        x = stats[i, cv2.CC_STAT_LEFT]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if area>500 and area<2500:
            if not (w/h > 2.5 or h/w > 2.5):
                img = np.zeros((1,50,50),dtype=np.float32)
                img[0,:,:] = cv2.resize(maskedImage[y-2:y+h+2,x-2:x+w+2], (50,50))
                prediction = recognitoinModel.predict(img)
                numberPlate+=mdl.character[np.argmax(prediction[0])]

    print(numberPlate)






    # numbers = np.array(numbers)
    # maxX,maxY,maxW,maxH = np.max(numbers, axis=0)
    # x,y,_,_ = np.min(numbers, axis=0)

    # numberPlate = grayImage[y-5:maxY+maxH+5,x-5:maxX+maxW+5]
    # if (numberPlate.shape[1]/numberPlate.shape[0]>2):
    #     numberPlate = cv2.resize(numberPlate, (300,75))
    # else:
    #     numberPlate = cv2.resize(numberPlate, (300,200))

