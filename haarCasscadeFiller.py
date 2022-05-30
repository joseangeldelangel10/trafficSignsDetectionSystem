import cv2 as cv
import numpy as np
import imutils
import os
import time

directory = "D:/tecdemty/6to Semestre/robotica inteligente/signsDetectionModel2/blackCircles/p"

cap = cv.VideoCapture(0)
x1, y1 = 200, 100
x2, y2 = 350, 250
count = 0
kernel = np.ones((5,5), np.uint8)

while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)    
    
    #red1 = cv.inRange(HSV,(170,70,50),(180,255,255))
    #red2 = cv.inRange(HSV,(0,70,50),(10,255,255))

    _,red = cv.threshold(gray, 128, 255, cv.THRESH_BINARY_INV)
    #red = cv.inRange(HSV,(0,0,0),(255,255,100))
    red = cv.dilate(red, kernel, iterations=1)

    if ret == False: break
    redCopy = red.copy()
    red = cv.merge((red, red, red))
    cv.rectangle(red,(x1,y1),(x2,y2),(255,0,0),2)
    objeto = redCopy[y1:y2,x1:x2]
    objeto = imutils.resize(objeto,width=48)
    #print(objeto.shape)
    k = cv.waitKey(1)
    if k == ord('s'):
        cv.imwrite(directory+'/objeto_{}.jpg'.format(str(time.time())),objeto)
        print('Imagen saved succesfully')
        count = count +1
    if k == 27:
        break
    cv.imshow('frame',red)
    cv.imshow('objeto',objeto)
cap.release()
cv.destroyAllWindows()