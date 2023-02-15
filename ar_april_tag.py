'''
Sample Usage:-
python pose_estimation.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --type DICT_5X5_100
'''


import numpy as np
import cv2
import sys
from utils import ARUCO_DICT
import argparse
import time

# objp = np.zeros((24*17,3), np.float32)
# objp[:,:2] = np.mgrid[0:24,0:17].T.reshape(-1,2)
# axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

# axisBoxes = np.float32([[-0.00,-0.00,0], [0,0.02,0], [0.02,0.02,0], [0.02,0,0],
#                    [0,0,0.02],[0,0.02,0.02],[0.02,0.02,0.02],[0.02,0,0.02] ])

axisBoxes = np.float32([[-0.01, -0.01, 0], [-0.01, 0.01, 0], [0.01, 0.01, 0], [0.01, -0.01, 0],
  					   [-0.01, -0.01, 0.02], [-0.01, 0.01, 0.02], [0.01, 0.01, 0.02],[0.01, -0.01, 0.02]])




# This function draws lines joining the given image points to the first chess board corner
# def draw(img, corners, imgPoints):
#     corner = tuple(corners[0].ravel())
#     img = cv2.line(img, corner, tuple(imgPoints[0].ravel()), (255, 0, 0), 5)
#     img = cv2.line(img, corner, tuple(imgPoints[1].ravel()), (0, 255, 0), 5)
#     img = cv2.line(img, corner, tuple(imgPoints[2].ravel()), (0, 0, 255), 5)
#     return img


def drawBoxes(img, corners, imgpts): 
    
    """
    This Funtion draws lines using the img points and return a image with lines in the image in homogeneous frame. 
    
    Inputs:
        _img_: The raw image on top of which we want to draw bounding boxes
        _corners_: The corner points of the boxes
        _imgpts_: The points at from where to where the lines should be drawn in Homogeneous coordinate

    Returns:
        _img_: output image with bounding boxes
    """

    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

    return img


def pose_esitmation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):

    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera
    return:-
    frame - The frame with the axis drawn on it
    '''

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()


    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict, parameters=parameters)

        # If markers are detected
    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients, distortion_coefficients)
            

            # Projected 3D points to image plane
            imgpts, jac = cv2.projectPoints(axisBoxes, rvec, tvec, matrix_coefficients, distortion_coefficients)

            box_img = drawBoxes(frame, corners, imgpts)
            #cv2.imshow('img', box_img)
            
            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners) 

            # Draw Axis
            #cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  

    return frame

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    # ap.add_argument("-k", "--K_Matrix", required=True, help="Path to calibration matrix (numpy file)")
    # ap.add_argument("-d", "--D_Coeff", required=True, help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")
    args = vars(ap.parse_args())

    
    if ARUCO_DICT.get(args["type"], None) is None:
        print(f"ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

    aruco_dict_type = ARUCO_DICT[args["type"]]
    
    k = np.load("calibration_matrix.npy")   # Imoprting K matrix
    d = np.load("distortion_coefficients.npy")  # Importing d matrix

    video = cv2.VideoCapture(0)
    time.sleep(2.0)

    while True:
        ret, frame = video.read()

        if not ret:
            break
        
        output = pose_esitmation(frame, aruco_dict_type, k, d)

        cv2.imshow('Estimated Pose', output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()