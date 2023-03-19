import cv2
import numpy as np

def distance(x,y):
    (x1,y1,_,_) = x
    (x2,y2,_,_) = y
    dx = x2 - x1
    dy = y2 - y1
    return dx*dx+dy*dy


if __name__=='__main__':
    id = 3
    image = cv2.imread(f"images/{id}.jpg")
    
    image = cv2.resize(image, (620,480) )
    # image = cv2.GaussianBlur(image, (3,3), 0)
    # imageHSV = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

    # lowerRed1 = np.array([0,100,20])
    # upperRed1 = np.array([10,255,255])
    # lowerRed2 = np.array([160,100,20])
    # upperRed2 = np.array([179,255,255])
    
    # lowerRedMask = cv2.inRange(imageHSV,lowerRed1,upperRed1)
    # upperRedMask = cv2.inRange(imageHSV,lowerRed2,upperRed2)

    # mask = lowerRedMask+upperRedMask
    
    # maskedImage = cv2.bitwise_and(image,image,mask=mask)
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _,maskedImage = cv2.threshold(grayImage,50,255,cv2.THRESH_OTSU)

    connections = cv2.connectedComponentsWithStats(maskedImage, 2, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = connections

    outputImage = image.copy()

    letterLike = []

    for i in range(1,numLabels):
        y = stats[i, cv2.CC_STAT_TOP]
        x = stats[i, cv2.CC_STAT_LEFT]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if (area > 50 and area < 500):
            if not((w/h)>1.5 or (h/w)>1.5):
                cv2.rectangle(outputImage, (x,y), (x+w,y+h), (0,255,0), 2)
                letterLike.append([x,y,w,h])


    # find cluster based on distance
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
                for k,letter in enumerate(letterLike):
                    if not tracked[k]:
                        if distance(letterLike[c[j]],letterLike[k]) < 1600:
                            tracked[k] = True
                            c.append(k)
                j+=1
        else:
            i += 1

    numberPlateCluster = cluster[np.argmax([len(i) for i in cluster])]
    numbers = []
    for c in numberPlateCluster:
        numbers.append(letterLike[c])

    numbers = np.array(numbers)
    maxX,maxY,maxW,maxH = np.max(numbers, axis=0)
    x,y,_,_ = np.min(numbers, axis=0)

    numberPlate = grayImage[y-5:maxY+maxH+5,x-5:maxX+maxW+5]
    if (numberPlate.shape[1]/numberPlate.shape[0]>2):
        numberPlate = cv2.resize(numberPlate, (300,75))
    else:
        numberPlate = cv2.resize(numberPlate, (300,200))

    cv2.imwrite(f"numberPlates/{id}.jpg", numberPlate)
