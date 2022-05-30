import cv2 as cv
import numpy as np
import imutils
import os
import time

directory = "C:/Users/super/Documents/6toSemestre/signDetectionCnn/0"

"""
codifing:
0 - Stop sign
1 - Straight ahead
2 - Roundobout
3 - turn right
4 - turn left
5 - Prohibition ends
6 - Semaphore red
7 - Semaphore yellow
8 - Semaphore green
"""

cap = cv.VideoCapture(0)
x1, y1 = 200, 100
x2, y2 = 350, 250
count = 0
while True:
    ret, frame = cap.read()

    #HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)    
    
    #red1 = cv.inRange(HSV,(170,70,50),(180,255,255))
    #red2 = cv.inRange(HSV,(0,70,50),(10,255,255))

    #red = cv.inRange(HSV,(0,0,0),(255,255,90))

    if ret == False: break
    frameCopy = frame.copy()
    cv.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
    objeto = frameCopy[y1:y2,x1:x2]
    objeto = imutils.resize(objeto,width=45)
    #print(objeto.shape)
    k = cv.waitKey(1)
    if k == ord('s'):
        cv.imwrite(directory+'/objeto_{}.jpg'.format(str(time.time())),objeto)
        print('Imagen saved succesfully')
        count = count +1
    if k == 27:
        break
    cv.imshow('frame',frame)
    cv.imshow('objeto',objeto)
cap.release()
cv.destroyAllWindows()