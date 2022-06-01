import os
from os import listdir
import cv2 as cv
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

cap = cv.VideoCapture(0)
kernel = np.ones((5,5), np.uint8)

red_hexagons_classifier = cv.CascadeClassifier("D:/tecdemty/6to Semestre/robotica inteligente/signsDetectionModel2/redHexagons/classifier/cascade.xml")
blue_circles_classifier = cv.CascadeClassifier("D:/tecdemty/6to Semestre/robotica inteligente/signsDetectionModel2/blueCircles/classifier/cascade.xml")
black_circles_classifier = cv.CascadeClassifier("D:/tecdemty/6to Semestre/robotica inteligente/signsDetectionModel2/blackCircles/classifier/cascade.xml")

cnn_model = load_model('traffic_signs_cnn_model_5/my_model')
cnn_probability_threshold = 0.90

def preprocessing(img):
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    img = np.reshape(img, (img.shape[0],img.shape[1],1))
    return img

def getCalssName(classNo):
    """if   classNo == 0: return 'Speed Limit 20 km/h'
    elif classNo == 1: return 'Speed Limit 30 km/h'
    elif classNo == 2: return 'Speed Limit 50 km/h'
    elif classNo == 3: return 'Speed Limit 60 km/h'
    elif classNo == 4: return 'Speed Limit 70 km/h'
    elif classNo == 5: return 'Speed Limit 80 km/h'
    elif classNo == 6: return 'End of Speed Limit 80 km/h'
    elif classNo == 7: return 'Speed Limit 100 km/h'
    elif classNo == 8: return 'Speed Limit 120 km/h'
    elif classNo == 9: return 'No passing'
    elif classNo == 10: return 'No passing for vechiles over 3.5 metric tons'
    elif classNo == 11: return 'Right-of-way at the next intersection'
    elif classNo == 12: return 'Priority road'
    elif classNo == 13: return 'Yield'
    elif classNo == 14: return 'Stop'
    elif classNo == 15: return 'No vechiles'
    elif classNo == 16: return 'Vechiles over 3.5 metric tons prohibited'
    elif classNo == 17: return 'No entry'
    elif classNo == 18: return 'General caution'
    elif classNo == 19: return 'Dangerous curve to the left'
    elif classNo == 20: return 'Dangerous curve to the right'
    elif classNo == 21: return 'Double curve'
    elif classNo == 22: return 'Bumpy road'
    elif classNo == 23: return 'Slippery road'
    elif classNo == 24: return 'Road narrows on the right'
    elif classNo == 25: return 'Road work'
    elif classNo == 26: return 'Traffic signals'
    elif classNo == 27: return 'Pedestrians'
    elif classNo == 28: return 'Children crossing'
    elif classNo == 29: return 'Bicycles crossing'
    elif classNo == 30: return 'Beware of ice/snow'
    elif classNo == 31: return 'Wild animals crossing'
    elif classNo == 32: return 'End of all speed and passing limits'
    elif classNo == 33: return 'Turn right ahead'
    elif classNo == 34: return 'Turn left ahead'
    elif classNo == 35: return 'Ahead only'
    elif classNo == 36: return 'Go straight or right'
    elif classNo == 37: return 'Go straight or left'
    elif classNo == 38: return 'Keep right'
    elif classNo == 39: return 'Keep left'
    elif classNo == 40: return 'Roundabout mandatory'
    elif classNo == 41: return 'End of no passing'
    elif classNo == 42: return 'End of no passing by vechiles over 3.5 metric tons'"""
    if   classNo == 0: return 'stop'
    elif classNo == 1: return 'go straigth'
    elif classNo == 2: return 'roundobout'
    elif classNo == 3: return 'turn right'
    elif classNo == 4: return 'turn left'
    elif classNo == 5: return 'end of prhibition'
    elif classNo == 6: return 'red semaphore'
    elif classNo == 7: return 'yellow semaphore'
    elif classNo == 8: return 'green semaphore' 

while True:
	_, originalFrame = cap.read()
	(originalWidth, originalHeight, _) = originalFrame.shape
	newWidth = int((7/10)*(originalWidth))
	newHeight = int((7/10)*(originalHeight))
	frame = cv.resize(originalFrame,(newHeight,newWidth), interpolation = cv.INTER_AREA)
	haarDetection = frame.copy()
	frameCopy = frame.copy()	
	HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)     
	
	red1 = cv.inRange(HSV,(145,70,50),(180,255,255))
	red2 = cv.inRange(HSV,(0,70,50),(10,255,255))
	red = cv.bitwise_or(red1,red2)	
	
	blue = cv.inRange(HSV,(85,70,50),(140,255,255))	

	#black = cv.inRange(HSV,(0,0,0),(255,255,100))
	black = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	_,black = cv.threshold(black, 128, 255, cv.THRESH_BINARY_INV)    
	black = cv.dilate(black, kernel, iterations=1)

	# TODO: ADD BLACK CIRCLES 	
		
	redHexagons = red_hexagons_classifier.detectMultiScale(red, scaleFactor=1.1, minNeighbors=350, minSize=(40,40))
	blueCircles = blue_circles_classifier.detectMultiScale(blue, scaleFactor=1.1, minNeighbors=700, minSize=(40,40))
	blackCircles = black_circles_classifier.detectMultiScale(black, scaleFactor=1.1, minNeighbors=300, minSize=(40,40))
	
	for (x,y,w,h) in redHexagons:
		cv.rectangle(haarDetection, (x,y),(x+w,y+h),(0,0,255),2)
		cv.putText(haarDetection,'redHexagon',(x,y-10),2,0.7,(0,0,255),2,cv.LINE_AA)

		red_hexagon_roi = frameCopy[y:y+h,x:x+w]
		red_hexagon_roi = cv.resize(red_hexagon_roi,(45,45), interpolation = cv.INTER_AREA)   
		red_hexagon_roi = preprocessing(red_hexagon_roi)
		red_hexagon_roi = red_hexagon_roi.reshape(1, 45, 45, 1)
		predictions = cnn_model.predict(red_hexagon_roi)	    
		classIndex = np.argmax(predictions)
		probabilityValue =np.amax(predictions)
		className = getCalssName(classIndex)
		if probabilityValue > cnn_probability_threshold and className == "stop":
			cv.rectangle(frameCopy, (x,y),(x+w,y+h),(0,0,255),2)
			cv.putText(frameCopy,'{class_name}, prob={prob:.2f}'.format(class_name=className, prob=probabilityValue),(x,y-10),2,0.7,(0,0,255),2,cv.LINE_AA)

	for (x,y,w,h) in blueCircles:
		cv.rectangle(haarDetection, (x,y),(x+w,y+h),(255,0,0),2)
		cv.putText(haarDetection,'blueCircle',(x,y-10),2,0.7,(255,0,0),2,cv.LINE_AA)

		blue_circle_roi = frameCopy[y:y+h,x:x+w]
		blue_circle_roi = cv.resize(blue_circle_roi,(45,45), interpolation = cv.INTER_AREA)   
		blue_circle_roi = preprocessing(blue_circle_roi)
		blue_circle_roi = blue_circle_roi.reshape(1, 45, 45, 1)
		predictions = cnn_model.predict(blue_circle_roi)	    
		classIndex = np.argmax(predictions)
		probabilityValue =np.amax(predictions)
		className = getCalssName(classIndex)
		if probabilityValue > cnn_probability_threshold and (className == "go straigth" or className == "roundobout" or className == "turn right" or className == "turn left"):
			cv.rectangle(frameCopy, (x,y),(x+w,y+h),(255,0,0),2)
			cv.putText(frameCopy,'{class_name}, prob={prob:.2f}'.format(class_name=className, prob=probabilityValue),(x,y-10),2,0.7,(255,0,0),2,cv.LINE_AA)


	for (x,y,w,h) in blackCircles:
		cv.rectangle(haarDetection, (x,y),(x+w,y+h),(0,0,0),2)
		cv.putText(haarDetection,'blackCircle',(x,y-10),2,0.7,(0,0,0),2,cv.LINE_AA)

		black_circle_roi = frameCopy[y:y+h,x:x+w]
		black_circle_roi = cv.resize(black_circle_roi,(45,45), interpolation = cv.INTER_AREA)   
		black_circle_roi = preprocessing(black_circle_roi)
		black_circle_roi = black_circle_roi.reshape(1, 45, 45, 1)
		predictions = cnn_model.predict(black_circle_roi)	    
		classIndex = np.argmax(predictions)
		probabilityValue =np.amax(predictions)
		className = getCalssName(classIndex)
		if probabilityValue > cnn_probability_threshold and (className == "end of prhibition"):
			cv.rectangle(frameCopy, (x,y),(x+w,y+h),(0,0,0),2)
			cv.putText(frameCopy,'{class_name}, prob={prob:.2f}'.format(class_name=className, prob=probabilityValue),(x,y-10),2,0.7,(0,0,0),2,cv.LINE_AA)

	cv.imshow('haarDetection',haarDetection)
	cv.imshow('final',frameCopy)	
	
	if cv.waitKey(1) == 27:
		break

cap.release()
cv.destroyAllWindows()