import numpy as np
np.random.seed(1337)
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
import cv2 as cv
import pickle
import matplotlib.pyplot as plt

#loading the data we previously gathered

pickle_in = open("X_train.pickle","rb")
X_train = pickle.load(pickle_in)

pickle_in = open("y_train.pickle","rb")
y_train = pickle.load(pickle_in)

X_train = X_train/255.0


n_classes = 3
# print("Shape before one-hot encoding: ", y_train)
Y_train = np_utils.to_categorical(y_train, n_classes)
# print("Shape after one-hot encoding: ", Y_train)

print(len(y_train))
print(len(X_train))

model = Sequential()

model.add(Conv2D(filters = 8, kernel_size = (3, 3), activation = 'relu', padding = 'same', input_shape = (192, 192, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides = 2))#1


model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation = 'relu', padding = 'same', input_shape = (192, 192, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides = 2))#2

model.add(tf.keras.layers.BatchNormalization())

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', padding = 'same', input_shape = (192, 192, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides = 2))#3

model.add(tf.keras.layers.BatchNormalization())

model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', padding = 'same', input_shape = (192, 192, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides = 2))#4

model.add(Flatten())

model.add(Dense(units = n_classes, activation = 'softmax'))

model.add(Dense(3))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=16, epochs=10, validation_split=0.2)


#Saving the model on my computer
model.save('CoinJump_Image_Model.h5', overwrite=True)

