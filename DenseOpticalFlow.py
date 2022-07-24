import cv2 as cv
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "1"



def dense_optical_flow(method, video_path, params=[], to_gray=False):
    # read the video
    cap = cv.VideoCapture(video_path)
    # Read the first frame
    ret, old_frame = cap.read()

    # crate HSV & make Value a constant
    hsv = np.zeros_like(old_frame)
    hsv[..., 1] = 255

    # Preprocessing for exact method
    if to_gray:
        old_frame = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    while True:
        # Read the next frame
        ret, new_frame = cap.read()
        frame_copy = new_frame
        if not ret:
            break
        # Preprocessing for exact method
        if to_gray:
            new_frame = cv.cvtColor(new_frame, cv.COLOR_BGR2GRAY)
        # Calculate Optical Flow
        flow = method(old_frame, new_frame, None, *params)

        # Encoding: convert the algorithm's output into Polar coordinates
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        # Use Hue and Saturation to encode the Optical Flow
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        # Convert HSV image into BGR for demo
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)


        # 이미지의 크기를 잡고 이미지의 중심을 계산합니다.
        (h1, w1) = frame_copy.shape[:2]
        (h2, w2) = bgr.shape[:2]

        (cX1, cY1) = (w1 // 2, h1 // 2)
        (cX2, cY2) = (w2 // 2, h2 // 2)

        # 이미지를 중심으로 -90도 회전
        M = cv.getRotationMatrix2D((cX1, cY1), -90, 1.0)
        frame_copy_rotate = cv.warpAffine(frame_copy, M, (w1, h1))
        cv.imshow(frame_copy_rotate)

        M = cv.getRotationMatrix2D((cX2, cY2), -90, 1.0)
        bgr_rotate = cv.warpAffine(bgr, M, (w2, h2))
        cv.imshow(bgr_rotate)
        
        k = cv.waitKey(25) & 0xFF
        if k == 27:
            break
        old_frame = new_frame

def main():
    video_path = "JumpTest2.mp4"
    method = cv.calcOpticalFlowFarneback
    params = [0.5, 3, 15, 3, 5, 1.2, 0]  # Farneback's algorithm parameters
    dense_optical_flow(method, video_path, params, to_gray=True)


if __name__ == "__main__":
    main()