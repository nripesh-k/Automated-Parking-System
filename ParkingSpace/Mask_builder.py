import cv2
import pickle
 
try:
    with open('CarParkPos.txt', 'rb') as f:
        posList = pickle.load(f)
except:
    posList = []
 
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

    # elif events == cv2.EVENT_LBUTTONUP:
    #     pt2 = (x,y)

    elif events == cv2.EVENT_MBUTTONDOWN:
        for i, pos in enumerate(posList):
            p1, p2 = pos
            if (p1[0] < x < p2[0] or p2[0] < x < p1[0]) and (p1[1]< y < p2[1] or p2[1]< y < p1[1]):
                posList.pop(i)
 
    with open('CarParkPos.txt', 'wb') as f:
        pickle.dump(posList, f)

while True:
    img = cv2.imread('carParkImg.png')
    for pos in posList:
        cv2.rectangle(img, pos[0], pos[1], (255, 0, 255), 1)
 
    cv2.imshow("RectangularSlots", img)
    cv2.setMouseCallback("RectangularSlots", mouseClick)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break