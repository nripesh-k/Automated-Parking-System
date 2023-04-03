import pickle

from skimage.transform import resize
import numpy as np
import cv2

MODEL = pickle.load(open("./Resources/model.p", "rb"))

def empty_or_not(spot_bgr):

    flat_data = []

    img_resized = resize(spot_bgr, (15, 15, 3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)

    y_output = MODEL.predict(flat_data)

    if y_output == 0:
        return True
    else:
        return False

def get_parking_spots(connected_components):
    (totalLabels, label_ids, values, centroid) = connected_components

    slots = []
    coef = 1
    for i in range(1, totalLabels):

        # This extracts the coordinates
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)

        slots.append([x1, y1, w, h])

    return slots

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