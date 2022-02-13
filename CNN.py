#Import the CNN librarie

from keras.models import Sequential
from keras.layers import Convolution2D,MaxPooling2D,Flatten,Dense

#Initilise the model

classifier = Sequential()

#Convolutional Layer
bit_depth = 3 #For Colur Image(r/g/b), For B/W = bit_Depth = 1

#Convolution2D Parameter layer
#1) No.of Filters
#2) Stride size(l and H)
#3) Image size

classifier.add(Convolution2D(32,(2,2),input_shape = (64,64, bit_depth),activation = "relu"))

#MaxPooling Layer
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Flatten
classifier.add(Flatten())

#Fully Connected Layer
#Hiddeen layer 1 = nodes = 8
classifier.add(Dense(units = pow(2,3),activation = "relu"))

#Hidden layer 2 = nodes = 4
classifier.add(Dense(units = pow(2,2),activation = "relu"))

#Output layer = nodes = 1
classifier.add(Dense(units = 1, activation = "sigmoid"))

#Compile the model 
classifier.compile(optimizer = "rmsprop",loss = "binary_crossentropy",metrics = ["accuracy"])


#Image Pre-Processing and image augmentation
from keras.preprocessing.image import ImageDataGenerator
#Apply the augmentation on train and validation image

train_datagen = ImageDataGenerator(rescale = 1/255,shear_range = 0.3,zoom_range = 0.4,horizontal_flip = True)

val_datagen = ImageDataGenerator(rescale = 1/255,shear_range = 0.3,zoom_range                                               = 0.4,horizontal_flip = True)

#set up the directry structure to reed the images

train_dir = "C:/Users/HP/Desktop/All program files/stat/binary/train"
val_dir = "C:/Users/HP/Desktop/All program files/stat/binary/validation"

#read the files

train_set = train_datagen.flow_from_directory(train_dir,target_size = (64,64),batch_size = 5,class_mode = "binary")


val_set =  val_datagen.flow_from_directory(val_dir,target_size = (64,64),batch_size = 5,class_mode = "binary")

#train model

EPOCHS = 30

classifier.fit(train_set,steps_per_epoch = 20,epochs = EPOCHS,validation_data =val_set,validation_steps = 20)


#Predict the test images
from keras.preprocessing import image
import numpy as np
import os

#test path 

test_dir = "C:/Users/HP/Desktop/All program files/stat/binary/test/"

#create a list tostore the file names of the test images
testimages = []

for p,d,files in os.walk(test_dir):
    for f in files:
        testimages.append(p+f)

print(testimages)    

#stack up the images fro prediction
imagestack = None
for i in testimages:
    img = image.load_img(i,target_size =(64,64) )
    #Converting image into array format
    y = image.img_to_array(img)
    y = np.expand_dims(y,axis=0)
    y /= 255 #rescale the image to match the image Generator setting
    
    if imagestack is None:
        imagestack = y
    else:
        imagestack = np.vstack([imagestack,y])
        
print(imagestack)

#predict

predy = classifier.predict_classes(imagestack)
predy =predy.reshape(-1)
predy

#try this 
classifier.predict(imagestack)








































