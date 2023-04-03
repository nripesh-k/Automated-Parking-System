import cv2
import time
import numpy as np
import model as mdl
import functions as fn

def predictExitNumberPlate(numberPlate):
    numberPlatePrediction = ''
    recognitoinModel = mdl.model()
    recognitoinModel.load_weights('trainedModel.h5')
    maskedNumberPlateImage = cv2.GaussianBlur(numberPlate,(3,3),0)
    _, maskedNumberPlateImage =  cv2.threshold(maskedNumberPlateImage,0,255,cv2.THRESH_OTSU)
    _,numberPlateThreshImage = cv2.threshold(numberPlate,0,255,cv2.THRESH_OTSU)
    kernel = np.ones((2,2))
    maskedNumberPlateImage = cv2.erode(maskedNumberPlateImage, kernel, iterations=1)
    numberPlateThreshImage = cv2.erode(numberPlateThreshImage, kernel, iterations=1)
    connections = cv2.connectedComponentsWithStats(maskedNumberPlateImage, 2, cv2.CV_32S)
    (numLabels, _, stats, _) = connections
    rectangleImage = maskedNumberPlateImage.copy()
    letters = []
    for i in range(1,numLabels):
        y = stats[i, cv2.CC_STAT_TOP]
        x = stats[i, cv2.CC_STAT_LEFT]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if not(area<150 or area>1234):
            if w/h<3 and h/w<3:
                letters.append([x,y,w,h])

    img = np.zeros((1,50,50),dtype=np.float32)
    letters.sort(key=lambda x: x[0])
    for l in letters:
        x,y,w,h = l
        img[0,:,:] = cv2.resize(numberPlateThreshImage[y-2:y+h+2,x-2:x+w+2], (50,50))
        rectangleImage = cv2.rectangle(rectangleImage, (x,y), (x+w,y+h), (255,255,255),2)
        prediction = recognitoinModel.predict(img, verbose=0)
        numberPlatePrediction+=mdl.character[np.argmax(prediction[0])]

    cv2.imshow('1', rectangleImage)
    cv2.waitKey(1000)
    return numberPlatePrediction


if __name__=='__main__':
    video = cv2.VideoCapture('../Blender/exit.mp4')

    waitTimer = None
    while True:
        if video.get(cv2.CAP_PROP_POS_FRAMES) == video.get(cv2.CAP_PROP_FRAME_COUNT):
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        success, image = video.read()
        min_width = 120
        image = cv2.resize(image, (620,420) )
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _,maskedImage = cv2.threshold(grayImage,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        cv2.imshow('exit', image)
        cv2.waitKey(25)

        if not waitTimer:
            connections = cv2.connectedComponentsWithStats(maskedImage, 2, cv2.CV_32S)
            (numLabels, labels, stats, centroids) = connections
            components = grayImage.copy()
            letterLike = []
            for i in range(1,numLabels):
                y = stats[i, cv2.CC_STAT_TOP]
                x = stats[i, cv2.CC_STAT_LEFT]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                area = stats[i, cv2.CC_STAT_AREA]
                if (area > 65 and area < 300):
                    if not((w/h)>2 or (h/w)>2):
                        letterLike.append([x,y,w,h])
            numbers = fn.findCluster(letterLike)

            if numbers:
                predict = False
                numbers = np.array(numbers)
                maxX,maxY,maxW,maxH = np.max(numbers, axis=0)
                x,y,_,_ = np.min(numbers, axis=0)
                numberPlate = []
                if (maxX+maxW-x)/(maxY+maxH-y)>2:
                    numberPlate = grayImage[y-5:maxY+maxH+5,x-30:maxX+maxW+30]
                    numberPlate = cv2.resize(numberPlate, (300,80))
                    localizedImage = cv2.rectangle(grayImage, (x,y), (maxX+maxW, maxY+maxH), (0,200,0), 2)
                    predict = True

                if predict:
                    numberPlatePrediction = predictExitNumberPlate(numberPlate)
                    print(f"{numberPlatePrediction} exited")
                    waitTimer = time.time()
        else:
            if time.time() - waitTimer > 3:
                waitTimer = None