import cv2
import pickle
import numpy as np
import time
import model as mdl
import util as fn

lot_mask = './Resources/mask_animated.png'
lot_path = '../Blender/main.mp4'
lot_mask = cv2.imread(lot_mask, 0)

# Video feeds
lot = cv2.VideoCapture(lot_path)
entry_video = cv2.VideoCapture('../Blender/entry.mp4')
exit_video = cv2.VideoCapture('../Blender/exit.mp4')
waitTimer1 = None
waitTimer2 = None

# Getting the boxes with connected components usage
connected_components = cv2.connectedComponentsWithStats(lot_mask, 4, cv2.CV_32S)
spots = fn.get_parking_spots(connected_components)

recognitionModel = mdl.NLPD_model()
recognitionModel.load_weights('./Resources/NLPD_Model.h5')

records = {}
rate = 9.5

def checkParkingSpace(imgPro, img):
    spaceCounter = 0

    for pos in spots:
        x1, y1, w, h = pos

        imgCrop = imgPro[y1:y1 + h, x1:x1 + w]
        count = cv2.countNonZero(imgCrop)

        if count < 200:
            color = (0, 255, 0)
            spaceCounter += 1
        else:
            color = (0, 0, 200)
        cv2.rectangle(img, (x1,y1-5), (pos[0] + pos[2], pos[1] + pos[3]), color, 2)
        # cv2.putText(img, str(count), (x1, y1 + h ), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        #                    color, 2)
        
    cv2.rectangle(img, (20, 20), (350, 60), (0, 0, 0), -1)
    cv2.putText(img, 'Available spots: {} / {}'.format(str(spaceCounter), str(len(spots))), (25, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

def lotMonitor():
    _, img = lot.read()

    img = cv2.resize(img,(630,732))        

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgMedian = cv2.medianBlur(imgGray, 5)
    imgThreshold = cv2.adaptiveThreshold(imgMedian, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 25, 16)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgThreshold, kernel, iterations=1)

    checkParkingSpace(imgDilate, img)
    cv2.imshow("Parking Lot", img)
    cv2.moveWindow("Parking Lot", 650,100)
    # cv2.imshow("imgThreshold", imgThreshold)
    # cv2.imshow("imgDilate", imgDilate)

def predictEntranceNumberPlate(numberPlate):
    global recognitionModel
    numberPlatePrediction = ''
    maskedNumberPlateImage = cv2.adaptiveThreshold(numberPlate,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,81,-2)
    # maskedNumberPlateImage = cv2.GaussianBlur(maskedNumberPlateImage,(3,3),0)
    _, maskedNumberPlateImage =  cv2.threshold(numberPlate,100,255,cv2.THRESH_BINARY)
    kernel = np.ones((2,2))
    maskedNumberPlateImage = cv2.erode(maskedNumberPlateImage, kernel, iterations=1)

    connections = cv2.connectedComponentsWithStats(maskedNumberPlateImage, 2, cv2.CV_32S)
    (numLabels, _, stats, _) = connections
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

    img = np.zeros((1,32,32),dtype=np.float32)
    letters.sort(key=lambda x: x[0])
    for l in letters:
        x,y,w,h = l
        try:
            # letterImage = maskedNumberPlateImage[y-2:y+h+2,x-2:x+w+2]
            letterImage = numberPlate[y-2:y+h+2,x-2:x+w+2]
            _, letterImage = cv2.threshold(letterImage,0,255,cv2.THRESH_OTSU)
            img[0,:,:] = cv2.resize(letterImage, (32,32))
            prediction = recognitionModel.predict(img, verbose=0)
            numberPlatePrediction+=mdl.NLPD_characters[np.argmax(prediction[0])]
        except:
            continue
    return numberPlatePrediction

def entranceMonitor():
    global waitTimer1
    _, image = entry_video.read()
    image = cv2.resize(image, (620,420) )
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshImage = cv2.adaptiveThreshold(grayImage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,401,-1)
    _,maskedImage = cv2.threshold(threshImage,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imshow('Entrance', image)
    cv2.moveWindow('Entrance', 20, 100)
    # cv2.waitKey(25)
    if not waitTimer1:
        connections = cv2.connectedComponentsWithStats(maskedImage, 2, cv2.CV_32S)
        (numLabels, _, stats, _) = connections
        # components = grayImage.copy()
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
                    # components = cv2.rectangle(components, (x,y), (x+w,y+h), (0,200,0), 2)
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
                predict = True

            if predict:
                numberPlatePrediction = predictEntranceNumberPlate(numberPlate)
                records[numberPlatePrediction] = time.time()
                print(f"{numberPlatePrediction} has entered")
                waitTimer1 = time.time()
    else:
        if time.time() - waitTimer1 > 3:
            waitTimer1 = None

def predictExitNumberPlate(numberPlate):
    global recognitionModel
    numberPlatePrediction = ''
    maskedNumberPlateImage = cv2.GaussianBlur(numberPlate,(3,3),0)
    _, maskedNumberPlateImage =  cv2.threshold(maskedNumberPlateImage,0,255,cv2.THRESH_OTSU)
    _,numberPlateThreshImage = cv2.threshold(numberPlate,0,255,cv2.THRESH_OTSU)
    kernel = np.ones((2,2))
    maskedNumberPlateImage = cv2.erode(maskedNumberPlateImage, kernel, iterations=1)
    numberPlateThreshImage = cv2.erode(numberPlateThreshImage, kernel, iterations=1)
    connections = cv2.connectedComponentsWithStats(maskedNumberPlateImage, 2, cv2.CV_32S)
    (numLabels, _, stats, _) = connections
    # rectangleImage = maskedNumberPlateImage.copy()
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

    img = np.zeros((1,32,32),dtype=np.float32)
    letters.sort(key=lambda x: x[0])
    for l in letters:
        x,y,w,h = l
        letterImage = numberPlateThreshImage[y-4:y+h+4,x-4:x+w+4]
        # _, letterImage = cv2.threshold(letterImage,0,255,cv2.THRESH_OTSU)
        # letterImage = cv2.erode(letterImage, kernel, iterations=3)

        img[0,:,:] = cv2.resize(letterImage, (32,32))
        prediction = recognitionModel.predict(img, verbose=0)
        numberPlatePrediction+=mdl.NLPD_characters[np.argmax(prediction[0])]

    return numberPlatePrediction

def exitMonitor():
    global waitTimer2, rate
    _, image = exit_video.read()
    image = cv2.resize(image, (620,420) )
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _,maskedImage = cv2.threshold(grayImage,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imshow('Exit', image)
    cv2.moveWindow('Exit', 1290, 100)
    # cv2.waitKey(25)

    if not waitTimer2:
        connections = cv2.connectedComponentsWithStats(maskedImage, 2, cv2.CV_32S)
        (numLabels, _, stats, _) = connections
        # components = grayImage.copy()
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
                # localizedImage = cv2.rectangle(grayImage, (x,y), (maxX+maxW, maxY+maxH), (0,200,0), 2)
                predict = True

            if predict:
                numberPlatePrediction = predictExitNumberPlate(numberPlate)
                print(f"{numberPlatePrediction} has exited.")
                #manual
                try:
                    price = int((time.time() - records[numberPlatePrediction]) * rate/10) * 10
                    del records[numberPlatePrediction]
                    print(f" Total Fee: Rs. {price}")
                except:
                    pass
                waitTimer2 = time.time()
    else:
        if time.time() - waitTimer2 > 9:
            waitTimer2 = None


if __name__ == "__main__":
    while True:
    
        if lot.get(cv2.CAP_PROP_POS_FRAMES) == lot.get(cv2.CAP_PROP_FRAME_COUNT):
            lot.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        if entry_video.get(cv2.CAP_PROP_POS_FRAMES) == entry_video.get(cv2.CAP_PROP_FRAME_COUNT):
            entry_video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if exit_video.get(cv2.CAP_PROP_POS_FRAMES) == exit_video.get(cv2.CAP_PROP_FRAME_COUNT):
            exit_video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        lotMonitor()

        entranceMonitor()
        exitMonitor()

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break