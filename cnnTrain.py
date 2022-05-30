# Main requirements 

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
import cv2
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense

from sklearn.preprocessing import LabelBinarizer

def preprocessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = np.reshape(img, (img.shape[0],img.shape[1],1))
    return img

def myModel(num_output_classes, images_shape = (45, 45, 1)):

	# Hyperparameter selection

    # Filters of the CNN

    no_Of_Filters=60

    # Shape of the filters used in the CNN

    size_of_Filter=(5,5)
    size_of_Filter2=(3,3)                                                                                                                                                                                                        
    # Tekes batches of 2x2 pixels and avg the

    size_of_pool=(2,2)

    # Nodes of the neural classifier

    no_Of_Nodes = 500

    # =============== ADDING LAYERS DEFINDED IN PRESENTATION =========


    model = Sequential()

    conv1 = Conv2D(no_Of_Filters, size_of_Filter, data_format='channels_last', input_shape=images_shape)
    model.add(conv1)    
    model.add(Activation("relu"))
    #print("first conv 2D layer output size is: " + conv1.shape)
    conv2 = Conv2D(no_Of_Filters, size_of_Filter)
    model.add(conv2)
    model.add(Activation("relu"))    
    #print("second conv 2D layer output size is: " + conv2.shape)

    pol1 = MaxPooling2D(pool_size=size_of_pool)
    model.add(pol1)
    #print("max poling 1st layer output size is: " + pol1.shape)

    conv3 = Conv2D(no_Of_Filters/2, size_of_Filter2)
    model.add(conv3)
    model.add(Activation("relu"))
    #print("third conv 2D layer output size is: " + conv3.shape)

    conv4 = Conv2D(no_Of_Filters/2, size_of_Filter2)
    model.add(conv4)
    model.add(Activation("relu"))
    #print("fourth conv 2D layer output size is: " + conv4.shape)

    pol2 = MaxPooling2D(pool_size=size_of_pool)
    model.add(pol2)
    #print("max poling 1st layer output size is: " + pol2.shape)

    # -----------------CONVOLUTION PART ENDS 
    model.add(Dropout(0.49))
    model.add(Flatten())
    model.add(Dense(no_Of_Nodes))
    model.add(Activation("relu"))
    model.add(Dropout(0.49))
    model.add(Dense(num_output_classes))
    model.add(Activation("softmax"))

    # TODO: Add layers as presented in the class to conform your CNN

    model.compile(optimizer=tf.keras.optimizers.RMSprop(),loss=tf.keras.losses.BinaryCrossentropy(),metrics=[tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.FalseNegatives()])                                                                                  
    
    return model

################# Parameters #####################
tf.config.set_visible_devices([], 'GPU')

path = "C:/Users/super/Documents/6toSemestre/signDetectionCnn" # folder with all the class folders
#labelFile = 'labels.csv' # file with all names of classes
batch_size_val=50  # how many to process together before updating the interanl parameters
steps_per_epoch_val=100 # we divide all our database in 10 bathces 
epochs_val=8
imageDimesions = (45,45,1)
testRatio = 0.2    # if 1000 images split will 200 for testing
validationRatio = 0.2 # if 1000 images 20% of remaining 800 will be 160 for validation
###################################################


############################### Importing of the Images
count = 0
images = []
classNo = []
myList = os.listdir(path)
print("Total Classes Detected:",len(myList))
noOfClasses=len(myList)
print("Importing Classes.....")

#Import names
for x in range (0,len(myList)):
    myPicList = os.listdir(path+"/"+str(count))
    for y in myPicList:
        curImg = cv2.imread(path+"/"+str(count)+"/"+y)
        resizedImg = cv2.resize(curImg,(45,45), interpolation = cv2.INTER_AREA)
        resizedImg = preprocessing(resizedImg)
        images.append(resizedImg)
        classNo.append(count)
    # print(count, end =" ")
    count +=1
print(" ")
#print("classNo is: {var}".format(var=classNo))
images = np.array(images)
classNo = np.array(classNo)

lb = LabelBinarizer()
labelsEncoded = lb.fit_transform(classNo)
#print("images shape is: {im}".format(im=images.shape))
#print("images shape is: {classes}".format(classes=classNo.shape))

###############################

############################### Split Data
X_train, X_test, y_train, y_test = train_test_split(images, labelsEncoded, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)

print("x_train shape: {shape}".format(shape=X_train.shape))
print("x_test shape: {shape}".format(shape=X_test.shape))
print("y_train shape: {shape}".format(shape=y_train.shape))
print("y_test shape: {shape}".format(shape=y_test.shape))

# X_train = ARRAY OF IMAGES TO TRAIN
# y_train = CORRESPONDING CLASS ID

###############################

############################### TRAIN
# Create model structure
model = myModel(num_output_classes = len(myList) ,images_shape = (45,45,1))
dataGen = ImageDataGenerator()
print(model.summary())
# Train the model
history=model.fit_generator(dataGen.flow(X_train,y_train,batch_size=batch_size_val),epochs=epochs_val,validation_data=(X_validation,y_validation),shuffle=1)
model.save('traffic_signs_cnn_model_7/my_model')

############################### PLOT training data
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.show()
score =model.evaluate(X_test,y_test,verbose=0)
print('Test Score:',score[0])
print('Test Accuracy:',score[1])