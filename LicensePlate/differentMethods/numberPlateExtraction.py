import cv2
import imutils
import numpy as np

if __name__=='__main__':
    i = 2
    image = cv2.imread(f"images/{i}.jpg")
    
    image = cv2.resize(image, (620,480) )
    image = cv2.bilateralFilter(image, 15, 15, 15)
    imageHSV = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    lowerRed = np.array([160,100,20])
    upperRed = np.array([179,255,255])
    redMask = cv2.inRange(imageHSV,lowerRed,upperRed)
    maskedImage = cv2.bitwise_and(image,image,mask=redMask)
    maskedImage = cv2.cvtColor(cv2.cvtColor(maskedImage, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
    edgeD = cv2.Canny(maskedImage, 25, 300)

    # cv2.imshow('image edge', maskedImage)
    # cv2.waitKey(1000)

    # cv2.imshow('image edge', edgeD)
    # cv2.waitKey(1000)
    contours=cv2.findContours(edgeD.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours,key=cv2.contourArea, reverse = True)[:10]
    screenCnt = None

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break

    mask = np.zeros(maskedImage.shape,np.uint8)

    new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
    # new_image = cv2.bitwise_and(imageHSV,imageHSV,mask=mask)
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = image[topx:bottomx+1, topy:bottomy+1]
    numberPlate = cv2.cvtColor(Cropped, cv2.COLOR_BGR2GRAY)
    numberPlate = cv2.resize(numberPlate, (300,75))
    cv2.imwrite(f'numberPlates/{i}.jpg',numberPlate)

    cv2.imshow('image masked', Cropped)
    cv2.waitKey(1000)

    numberPlate = cv2.Canny(numberPlate, 150, 300)
    
    cv2.imshow('number plate', numberPlate)
    cv2.waitKey(2000)

    contours = cv2.findContours(numberPlate.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    letters = imutils.grab_contours(contours)
    letter = []
    for l in letters:
        mask = np.zeros(numberPlate.shape,np.uint8)
        peri = cv2.arcLength(l, True)
        approx = cv2.approxPolyDP(l, 0.3 * peri, True)
        if len(approx) == 4:
            cv2.drawContours(mask, [approx], 0, 255,-1)
            (x, y) = np.where(mask == 255)
            (topx, topy) = (np.min(x), np.min(y))
            (bottomx, bottomy) = (np.max(x), np.max(y))
            # Cropped = image[topx:bottomx+1, topy:bottomy+1]
            letter.append([topx,bottomx,topy,bottomy])
    
    letter.sort(key=lambda x: x[0])
    for e,l in enumerate(letter):
        letterImage = numberPlate[l[0]:l[1]+1,l[2]:l[3]+1]
        cv2.imwrite(f"numberPlates/{i}_{e}.jpg", letterImage)
