import cv2
import time
import numpy as np
import model as mdl

def distance(x,y):
    (x1,y1,_,_) = x
    (x2,y2,_,_) = y
    dx = x2 - x1
    dy = y2 - y1
    return dx*dx+dy*dy

def findCluster(letterLike):
    tracked = {}
    for i,_ in enumerate(letterLike):
        tracked[i] = False
    cluster = []
    i=0
    while (i<len(letterLike)):
        if not tracked[i]:
            cluster.append([])
            c = cluster[-1]
            c.append(i)
            tracked[i] = True
            i+=1
            j = 0
            while(j<len(c)):
                for k,_ in enumerate(letterLike):
                    if not tracked[k]:
                        if distance(letterLike[c[j]],letterLike[k]) < 500:
                            tracked[k] = True
                            c.append(k)
                j+=1
        else:
            i += 1
    if len(cluster) == 0:
        numberPlateCluster = []
    elif len(cluster) == 1:
        numberPlateCluster = cluster[0]
    else:
        numberPlateCluster = cluster[np.argmax([len(i) for i in cluster])]
    numbers = []
    # print(numberPlateCluster)
    if len(numberPlateCluster)>=6:
        for c in numberPlateCluster:
            numbers.append(letterLike[c])
    return numbers

def predictNumberPlate(numberPlate):
    numberPlatePrediction = ''
    recognitoinModel = mdl.model()
    recognitoinModel.load_weights('trainedModel.h5')

    # maskedNumberPlateImage = cv2.adaptiveThreshold(numberPlate,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,-2)
    # maskedNumberPlateImage = cv2.GaussianBlur(numberPlate,(3,3),0)
    _, maskedNumberPlateImage =  cv2.threshold(numberPlate,140,255,cv2.THRESH_BINARY)
    # kernel = np.ones((2,2))
    # maskedNumberPlateImage = cv2.erode(maskedNumberPlateImage, kernel, iterations=2)
    connections = cv2.connectedComponentsWithStats(maskedNumberPlateImage, 2, cv2.CV_32S)
    (numLabels, _, stats, _) = connections
    # rectangleImage = maskedNumberPlateImage.copy()
    letters = []
    for i in range(1,numLabels):
        y = stats[i, cv2.CC_STAT_TOP]
        x = stats[i, cv2.CC_STAT_LEFT]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        # area = stats[i, cv2.CC_STAT_AREA]
        letters.append([x,y,w,h])

    img = np.zeros((1,50,50),dtype=np.float32)
    letters.sort(key=lambda x: x[0])
    for l in letters:
        x,y,w,h = l
        img[0,:,:] = cv2.resize(maskedNumberPlateImage[y-4:y+h+4,x-4:x+w+4], (50,50))
        # rectangleImage = cv2.rectangle(rectangleImage, (x,y), (x+w,y+h), (255,255,255),2)
        prediction = recognitoinModel.predict(img, verbose=0)
        numberPlatePrediction+=mdl.character[np.argmax(prediction[0])]

    # cv2.imshow('1', rectangleImage)
    # cv2.waitKey(3000)
    return numberPlatePrediction


if __name__=='__main__':
    video = cv2.VideoCapture('example.mp4')
    entry = {}
    exit = {}
    priceCharged = {}
    
    while True:
        if video.get(cv2.CAP_PROP_POS_FRAMES) == video.get(cv2.CAP_PROP_FRAME_COUNT):
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        success, image = video.read()
        min_width = 120
        # image = cv2.imread("images/3.jpg")
        image = cv2.resize(image, (620,480) )
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _,maskedImage = cv2.threshold(grayImage,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        cv2.imshow('1', image)
        cv2.waitKey(25)
        connections = cv2.connectedComponentsWithStats(maskedImage, 2, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = connections
        letterLike = []
        for i in range(1,numLabels):
            y = stats[i, cv2.CC_STAT_TOP]
            x = stats[i, cv2.CC_STAT_LEFT]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            if (area > 70 and area < 300):
                if not((w/h)>2 or (h/w)>2):
                    letterLike.append([x,y,w,h])
        numbers = findCluster(letterLike)
        
        if numbers:
            predict = False
            numbers = np.array(numbers)
            maxX,maxY,maxW,maxH = np.max(numbers, axis=0)
            x,y,_,_ = np.min(numbers, axis=0)
            numberPlate = []
            if (maxX+maxW-x)/(maxY+maxH-y)>2:
                # if min_width <= maxX+maxW-x:
                numberPlate = grayImage[y-2:maxY+maxH+2,x-2:maxX+maxW+2]
                numberPlate = cv2.resize(numberPlate, (300,75))
                localizedImage = cv2.rectangle(image, (x,y), (maxX+maxW, maxY+maxH), (0,200,0), 2)
                cv2.imshow('1', localizedImage)
                cv2.waitKey(50)
                predict = True
            # else:
            #     if min_width//2 <= maxX+maxW-x:
            #         numberPlate = grayImage[y-2:maxY+maxH+2,x-2:maxX+maxW+2]
            #         numberPlate = cv2.resize(numberPlate, (300,200))
            #         predict = True

            if predict:
                numberPlatePrediction = predictNumberPlate(numberPlate)
                entered = entry.get(numberPlatePrediction)
                exiting = exit.get(numberPlatePrediction)

                if exiting:
                    if time.time() - exiting > 15:
                        del exit[numberPlatePrediction]

                if not entered and not exiting:
                    entry[numberPlatePrediction] = time.time()
                    print(f"{numberPlatePrediction} entered")
                elif entered: # entrance time-limit
                    if time.time() - entered < 15:
                        continue
                    else:
                        charge = (time.time() - entered)*4 # time*rate
                        priceCharged[numberPlatePrediction] = charge
                        del entry[numberPlatePrediction]
                        exit[numberPlatePrediction] = time.time()
                        print(f"{numberPlatePrediction} exited, Fee: Rs. {int(charge)}")