import sys
import cv2 as cv
import numpy as np

kernel = np.ones((8,8),np.uint8)
smaller_kernel = np.ones((4,4),np.uint8)

# define a video capture object
video = cv.VideoCapture(0)
  
while(True):
      
    # Capture the video frame by frame
    ret, frame = video.read()
    cv.imshow("Original image", frame.copy())

    HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)    
    
    red1 = cv.inRange(HSV,(170,70,50),(180,255,255))
    red2 = cv.inRange(HSV,(0,70,50),(10,255,255))

    red = cv.bitwise_or(red1,red2)

    blue = cv.inRange(HSV,(85,70,50),(140,255,255))

    black = cv.inRange(HSV,(0,0,0),(255,255,90))
    
    cv.imshow("red image", red)
    cv.imshow("blue image", blue)
    cv.imshow("black image", black)    
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the capture object
video.release()
# Destroy all the windows
cv.destroyAllWindows()