import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import numpy as np
import pickle

data_dir = '/Users/hsm/Documents/CalorieCoin/CoinJump/ImagePJ/HSM_FINAL/labelList'
labelList = ['Down', 'Land','Up']
for label in labelList:
    path = os.path.join(data_dir,label)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img))
        img_size = 192
        new_array = cv2.resize(img_array,(img_size,img_size))
        plt.imshow(new_array)

training_data = []

def create_trainSet():
    for label in labelList:
        path = os.path.join(data_dir,label)
        class_num = labelList.index(label)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img))
            resized_img_array = cv2.resize(img_array,(img_size,img_size))
            training_data.append([resized_img_array,class_num])

create_trainSet()

print(len(training_data))


random.seed(58)

#shuffling the training data to
#guarantee that the model will be well-trained

random.shuffle(training_data)

#appending the training data to a list using pickle
#X_train will be the images I will use to train the model
#and y_train will be the the 'category' that each image is in


X_train = []
y_train = []
for image,label in training_data:
    X_train.append(image)
    y_train.append(label)
X_train = np.array(X_train).reshape(-1, img_size, img_size,3)
pickle_out = open('X_train.pickle','wb')
pickle.dump(X_train, pickle_out)
pickle_out.close()

pickle_out = open('y_train.pickle','wb')
pickle.dump(y_train, pickle_out)
pickle_out.close()

