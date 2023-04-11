import cv2
import pickle
import numpy as np

from util import get_parking_spots

try: 
    posList = []
    mask = './Resources/mask.png'
    mask = cv2.imread(mask, 0)
    connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    spots = get_parking_spots(connected_components)
    for spot in spots:
        p1, p2 = (spot[0], spot[1]), (spot[2] + spot[0], spot[3] + spot[1])
        posList.append((p1, p2))
    print(posList)
except:
    posList = []

# try:
#     with open('CarParkPos.txt', 'rb') as f:
#         posList = pickle.load(f)
# except:
#     posList = []
 
pt1 = None
pt2 = None
count=0

def mouseClick(events, x, y, flags, params):

    global pt1, pt2, count, img

    if events == cv2.EVENT_LBUTTONDOWN:
        if count == 0:
            pt1 = (x,y)
            count = 1
        elif count == 1:
            pt2 = (x,y)
            posList.append((pt1, pt2))
            count = 0

    elif events == cv2.EVENT_MBUTTONDOWN:
        for i, pos in enumerate(posList):
            p1, p2 = pos
            if (p1[0] < x < p2[0] or p2[0] < x < p1[0]) and (p1[1]< y < p2[1] or p2[1]< y < p1[1]):
                posList.pop(i)
 
    # with open('CarParkPos.txt', 'wb') as f:
    #     pickle.dump(posList, f)

def create_mask():
    mask = np.zeros(img.shape[:2], dtype="uint8")
    for pos in posList:
        cv2.rectangle(mask, pos[0], pos[1], 255, -1)
    cv2.imshow('Mask', mask)
    cv2.imwrite('./Resources/mask.png',mask)

while True:
    img = cv2.imread('./Resources/lot.png')
    for pos in posList:
        cv2.rectangle(img, pos[0], pos[1], (0, 0, 255), 2)
 
    cv2.imshow("RectangularSlots", img)

    # Mouse Callback
    cv2.setMouseCallback("RectangularSlots", mouseClick)

    # Keyboard
    if cv2.waitKey(25) & 0xFF == ord('s'):
        create_mask()

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break