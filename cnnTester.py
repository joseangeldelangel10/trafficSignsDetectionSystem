import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import time

tf.config.set_visible_devices([], 'GPU')
 
#############################################
 
frameWidth= 640         # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.95         # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################
 
# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
"""cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)"""

model = load_model('traffic_signs_cnn_model_7/my_model')
 
def preprocessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
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
 
    # READ IMAGE
    success, imgOrignal = cap.read()
    #print("original image shape is {shape}".format(shape=imgOrignal.shape))
    imHeight, imWidth, _ = imgOrignal.shape
    imgOrignal = imgOrignal[:, int(imWidth/2 - imHeight/2):int(imWidth/2 + imHeight/2)]
         
    # preprocessing ESS IMAGE
    img = cv2.resize(imgOrignal,(45,45), interpolation = cv2.INTER_AREA)   
    img = preprocessing(img)
    img = img.reshape(1, 45, 45, 1)
    cv2.putText(imgOrignal, "CLASS: " , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    # PREDICT IMAGE
    predictions = model.predict(img)
    print("predinctions: " + str(predictions))
    classIndex = np.argmax(predictions)
    #classIndex = 0    
    probabilityValue =np.amax(predictions)
    if probabilityValue > threshold:
       print(getCalssName(classIndex))
       cv2.putText(imgOrignal,str(classIndex)+" "+str(getCalssName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
       cv2.putText(imgOrignal, str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
       cv2.imshow("Result", imgOrignal)
    else:
        print('Not found')
        cv2.putText(imgOrignal, 'No Traffic Sign', (120, 35), font, 0.75,
                    (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Result", imgOrignal)
         
    if cv2.waitKey(1) and 0xFF == ord('q'):
       break

    #time.sleep(2)