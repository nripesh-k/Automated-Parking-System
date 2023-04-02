import cv2
import pickle
import cvzone
import numpy as np
from util import get_parking_spots


mask = './mask_animated.png'

video_path = './main.mp4'

mask = cv2.imread(mask, 0)

# Video feed
cap = cv2.VideoCapture(video_path)
 
# Getting the boxes with connected components usage
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
posList = get_parking_spots(connected_components)

def checkParkingSpace(imgPro):
    spaceCounter = 0
 
    for pos in posList:
        x1, y1, w, h = pos
 
        imgCrop = imgPro[y1:y1 + h, x1:x1 + w]
        count = cv2.countNonZero(imgCrop)
 
        if count < 220:
            color = (0, 255, 0)
            spaceCounter += 1
        else:
            color = (0, 0, 200)
        cv2.rectangle(img, (x1,y1), (pos[0] + pos[2], pos[1] + pos[3]), color, 2)
        cvzone.putTextRect(img, str(count), (x1, y1 + h ), scale=1,
                           thickness=2, offset=0, colorR=color)

while True:
 
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success, img = cap.read()

    img = cv2.resize(img,(630,732))

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)
 
    checkParkingSpace(imgDilate)
    cv2.imshow("Parking Spot", img)
    # cv2.imshow("ImageBlur", imgBlur)
    # cv2.imshow("ImageThres", imgMedian)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break