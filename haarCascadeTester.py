import os
from os import listdir
import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
kernel = np.ones((5,5), np.uint8)
classifier = cv.CascadeClassifier("D:/tecdemty/6to Semestre/robotica inteligente/signsDetectionModel2/blackCircles/classifier/cascade.xml")


while True:
	_, originalFrame = cap.read()
	(originalWidth, originalHeight, _) = originalFrame.shape
	newWidth = int((7/10)*(originalWidth))
	newHeight = int((7/10)*(originalHeight))
	frame = cv.resize(originalFrame,(newHeight,newWidth), interpolation = cv.INTER_AREA)
	HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)     
	black = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	_,black = cv.threshold(black, 128, 255, cv.THRESH_BINARY_INV)    
	black = cv.dilate(black, kernel, iterations=1)

	#blue = cv.inRange(HSV,(85,70,50),(140,255,255))
	
	#red1 = cv.inRange(HSV,(145,70,50),(180,255,255))
	#red2 = cv.inRange(HSV,(0,70,50),(10,255,255))
	#red = cv.bitwise_or(red1,red2)
	
	#sign = classifier.detectMultiScale(black, scaleFactor=1.1, minNeighbors=300, minSize=(40,40))
	"""for (x,y,w,h) in sign:
		cv.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
		cv.putText(frame,'semaphore',(x,y-10),2,0.7,(0,255,0),2,cv.LINE_AA)"""
	cv.imshow('original frame', originalFrame)
	cv.imshow('frame',frame)
	cv.imshow('black',black)
	
	if cv.waitKey(1) == 27:
		break

cap.release()
cv.destroyAllWindows()