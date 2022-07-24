from datetime import datetime,timedelta
import time
import os
import requests
import cv2 as cv
import numpy as np

# save frame image
def saveFrame(frame, count, fileNum, frame_type = 'processed'):
    if frame_type == 'Original':
        savepath= r'/Users/hsm/Documents/CalorieCoin/CoinJump/GetFrame/Original'
        image_file_name = '{}.png'.format(count)
        save_path = os.path.join(savepath,image_file_name)
        cv.imwrite(save_path, frame)
    elif frame_type == 'processed':
        savepath= r'/Users/hsm/Documents/CalorieCoin/CoinJump/GetFrame/Change'
        image_file_name = '{}.png'.format(count)
        save_path = os.path.join(savepath,image_file_name)
        cv.imwrite(save_path, frame)
    else:
        savepath= r'/Users/hsm/Documents/CalorieCoin/CoinJump/GetFrame/Algorithm'
        image_file_name = '{}.png'.format(fileNum)
        save_path = os.path.join(savepath,image_file_name)
        cv.imwrite(save_path, frame)
        
# Get the frame from jumprope video by dense_optical_flow Algorithm
def getFrame_FDOF(method, video_path, params=[]):

    counter = 1
    fileNum = 1
    dims = (400,400)

    # read the video
    capture = cv.VideoCapture(video_path)

    # read the first frame
    ret, oldOriginal = capture.read()

    # Create HSV & make Value a constant
    hsv = np.zeros_like(oldOriginal)
    hsv[...,1] = 255

    # Preprocessing for exact method
    # change old frame to grayScale
    oldGray = cv.cvtColor(oldOriginal,cv.COLOR_BGR2GRAY)
    while(1):
        # Read the next frame
        flag, newOriginal = capture.read()
        if flag:
            newGray = cv.cvtColor(newOriginal, cv.COLOR_BGR2GRAY)
            
            # Calculate Optical Flow
            flow = method(oldGray, newGray, None, *params)
            
            # Encoding: convert the algorithm's output into Polar coordinates
            mag,angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Use Hue and Saturation to encode the Optical Flow
            hsv[...,0] = angle * 180 / np.pi/2
            hsv[...,2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
            
            # Convert HSV image into BGR for demo
            changed = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
            
            # Original image rotation & save
            resized_oldOriginal = cv.resize(oldOriginal, dims, interpolation=cv.INTER_AREA)
            (h1, w1) = resized_oldOriginal.shape[:2]
            (X1, Y1) = (w1 // 2, h1 // 2)
            M = cv.getRotationMatrix2D((X1, Y1), -180, 1.0)
            resized_oldOriginal_rotate = cv.warpAffine(resized_oldOriginal, M, (w1, h1))
            saveFrame(frame = resized_oldOriginal_rotate, count = counter, fileNum = fileNum, frame_type = 'Original')  
           

            # Changed image rotation & save
            resized_changed = cv.resize(changed, dims, interpolation=cv.INTER_AREA)
            (h2, w2) = resized_changed.shape[:2]
            (X2, Y2) = (w2 // 2, h2 // 2)
            M = cv.getRotationMatrix2D((X2, Y2), -180, 1.0)
            resized_changed_rotate = cv.warpAffine(resized_changed, M, (w2, h2))
            saveFrame(frame = resized_changed_rotate, count = counter, fileNum = fileNum, frame_type = 'processed')

            # Checking the progress
            if counter == 195:
                saveFrame(frame = oldOriginal, count = counter, fileNum = fileNum, frame_type = 'algorithm')
                fileNum+=1
                saveFrame(frame = oldGray, count = counter, fileNum = fileNum, frame_type = 'algorithm')
                fileNum+=1
                saveFrame(frame = newOriginal, count = counter, fileNum = fileNum, frame_type = 'algorithm')
                fileNum+=1
                saveFrame(frame = newGray, count = counter, fileNum = fileNum, frame_type = 'algorithm')
                fileNum+=1
                saveFrame(frame = resized_changed_rotate, fileNum = fileNum, count = counter, frame_type = 'algorithm')
                fileNum+=1
                print(flow)

            
            oldGray = newGray

            # get capture
            if cv.waitKey(20) & 0xFF == ord('x'):
                break
            counter += 1
        else:
            #total frame
            print(counter)
            break

def main():
    video_file_name = 'JumpTest2.mp4'
    folderDirectory = r"/Users/hsm/Documents/CalorieCoin/CoinJump/TestVideo/"
    fullDirectory = folderDirectory + video_file_name
    method = cv.calcOpticalFlowFarneback
    #params = {'pyr_scale':0.5,'levels':3, 'winsize':15,'iterations': 3, 'poly_n': 5,'poly_sigma':1.2,'flags':cv.OPTFLOW_USE_INITIAL_FLOW}
    params = [0.5, 3, 15, 3, 5, 1.2, 0]
    getFrame_FDOF(method, fullDirectory, params)

if __name__ == "__main__":
    main()