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
import time

# load model
model = tf.keras.models.load_model('CoinJump_Image_TM_Model.h5', compile=False)

def counting():
    
    jumpCnt = 0
    downCnt = 0
    upCnt = 0
    lastAction = 'None'

    cv.namedWindow('COIN JUMP')
    frame = cv.VideoCapture(0)
    oldRes, oldFrame = frame.read()
    cv.imshow("COIN JUMP", oldFrame)
    #flow = np.zeros((oldFrame.shape[0], oldFrame.shape[1], 2), dtype=np.float32)
    params4farne = {'pyr_scale':0.5,'levels':3, 'winsize':15,'iterations': 3, 'poly_n': 5,'poly_sigma':1.2,'flags':cv.OPTFLOW_USE_INITIAL_FLOW}
    hsv = np.zeros_like(oldFrame)
    hsv[...,1] = 255
    oldGray = cv.cvtColor(oldFrame,cv.COLOR_BGR2GRAY)
    #cv.imshow("COIN JUMP", oldFrame)

    while(1):
        newRes, newFrame = frame.read()

        stime = time.time()

        if newRes:
            newGray = cv.cvtColor(newFrame,cv.COLOR_BGR2GRAY)
            flow = cv.calcOpticalFlowFarneback(oldGray,newGray,None,0.5, 3, 15, 3, 5, 1.2, 0)
            mag,angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[...,0] = angle * 180 / np.pi/2
            hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
            flowImg = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
            dims = (400,400)
            resized_flowImg = cv.resize(flowImg,dims,interpolation=cv.INTER_AREA)
            resized_flowImg = cv.resize(resized_flowImg, (192, 192))
            resized_flowImg = resized_flowImg.reshape((1,) + resized_flowImg.shape)
            resized_flowImg = resized_flowImg/255.0
            flowPredict = model.predict(resized_flowImg)
            flowPredict = np.argmax(flowPredict, axis=-1)[0]
            
            if flowPredict == 0:
                lastAction = 'down'
                downCnt += 1
            elif flowPredict == 1:
                downCnt = 0
                if lastAction == 'down':
                    if upCnt>1:
                        jumpCnt += 1
                lastAction = 'land'
                upCnt = 0
            elif flowPredict == 2:
                if downCnt > 3:
                    jumpCnt += 1
                if lastAction == 'land' or lastAction == 'down':
                    upCnt = 1
                else:
                    upCnt += 1
                downCnt = 0
                lastAction = 'up'

            text = 'Jump : {}'.format(jumpCnt)
            font = cv.FONT_HERSHEY_SIMPLEX
            fontscale = 2
            fontColor = (255,255,255)
            textloc = (350,250)
            lineType = 2
            cv.putText(newFrame, text, textloc, font, fontscale, fontColor, lineType)
            oldGray = newGray
        else:
            #once the video is done processing
            #the function also returns the total amount of 
            #jumps a user has performed
            print('done')
            print(jumpCnt)
            break
        cv.resizeWindow("enhanced", 640, 480)
        
        etime = time.time()

        cv.imshow(f"COIN JUMP {etime - stime:.5f} sec", newFrame)
        cv.waitKey(1)
        #time.sleep(1)
        cv.destroyAllWindows()
    
    
                
counting()