import cv2
import imutils
import numpy as np

if __name__=='__main__':
    i = 2
    original = cv2.imread(f'numberPlates/{i}.jpg', cv2.IMREAD_GRAYSCALE)
    original = original[10:-10,:]
    numberPlate = cv2.GaussianBlur(original, (5,5), 0)
    # numberPlate = cv2.medianBlur(numberPlate, 3)
    numberPlate = cv2.adaptiveThreshold(numberPlate,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,-2)

    kernel = np.ones((2,2), np.uint8)
    numberPlate = cv2.erode(numberPlate, kernel, iterations=1)
    numberPlate = cv2.dilate(numberPlate, kernel, iterations=1)
    numberPlate = cv2.Canny(numberPlate, 1, 1)
    # numberPlate = cv2.Canny(numberPlate, 150, 300)
    
    cv2.imshow('number plate', numberPlate)
    cv2.waitKey(2000)

    contours = cv2.findContours(numberPlate.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    letters = imutils.grab_contours(contours)
    letter = []
    for l in letters:
        mask = np.zeros(numberPlate.shape,np.uint8)
        peri = cv2.arcLength(l, True)
        approx = cv2.approxPolyDP(l, 0.05 * peri, True)

        cv2.drawContours(mask, [approx], 0, 255,-1)
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        # Cropped = image[topx:bottomx+1, topy:bottomy+1]
        letter.append([topx,bottomx,topy,bottomy])
    
    letter.sort(key=lambda x: x[2])
    e = 0
    temp = letter[0][2]-30
    for l in letter:
        if (l[1]-l[0])*(l[3]-l[2])>500:
            if l[2]-20>temp:
                temp = l[2]
                e+=1
                letterImage = original[l[0]-5:l[1]+1+5,l[2]-5:l[3]+1+5]
                letterImage = cv2.resize(letterImage, (50,50))
                letterImage = cv2.adaptiveThreshold(letterImage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,-2)
                cv2.imwrite(f"numberPlates/{i}/{i}_{e}.jpg", letterImage)
