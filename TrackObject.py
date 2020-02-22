# Finished: Can track color with cam
# ToDo: Give unique ID? Publish network tables, connect from FRCVision to RoboRIO, Do contours for more accurate tracking
# Required dependencies are numpy and opencv-python
# Do pip3 install numpy opencv-python
# or ctrl + alt + s, select python interpreter, alt + enter, find numpy and opencv-python
# Keep in mind, cv2 is opencv and will not work if Pycharm tries to install cv2 dependency
# A question that I have in mind is that will the Roborio have opencv dependencies setup?

from __future__ import print_function
from math import pi,atan2,asin
import numpy as np
import time
import cv2


def nothing(x):
    pass


# HSV Notes
# H - Hue (Dominant Wavelength)
# S - Saturation (Purity / Shade of the color)
# V - Value (Intensity)

# ------Current "good" tracking settings that I found --------#
# Lower Hue = 69
# Lower Saturation = 69
# Lower Value = 69

# Upper Hue = 97
# Upper Saturation = 239
# Upper Value = 255

# Change according to which camera you want to use
# For laptops it should probably be 1
# Although in the pi it should probably be 0
# We can have a fail safe so that for sure there is a camera stream
cap = cv2.VideoCapture(1)

# Create a frame that has sliders to calibrate value
cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)

font = cv2.FONT_HERSHEY_COMPLEX

# Filler numpy array for the coordinates of the corners of the picture uwu
capturedTargetPoints = np.array([
                            (0, 0),     # Top left corner
                            (0, 0),     # Top right corner
                            (0, 0),     # Bottom left corner
                            (0, 0)      # Bottom right corner
                        ])

# 3D coordinates of target with arbitrary frame. Took these numbers from MS paint lol
# Z-axis can be zero for all of them because the tape is flat uwu
realTargetPoints = np.array([
                            (41.0, 193.0, 0.0),     # Top left corner
                            (350.0, 193.0, 0.0),    # Top right corner
                            (131.0, 327.0, 0.0),    # Bottom left corner
                            (273.0, 327.0, 0.0)     # Bottom right corner
                        ])

while True:
    # Maybe I want to put this into a function as well (Would that slow down the program?)
    # Get the values of the lower limit of color
    l_h = cv2.getTrackbarPos("LH", "Tracking")
    l_s = cv2.getTrackbarPos("LS", "Tracking")
    l_v = cv2.getTrackbarPos("LV", "Tracking")

    # Get the values of the upper limit of color
    u_h = cv2.getTrackbarPos("UH", "Tracking")
    u_s = cv2.getTrackbarPos("US", "Tracking")
    u_v = cv2.getTrackbarPos("UV", "Tracking")

    # This will be dynamic for now for calibration purposes
    # Define the lower color range
    l_b = np.array([l_h, l_s, l_v])
    # Define the upper color range
    u_b = np.array([u_h, u_s, u_v])

    # Read the frame from the cam
    _, frame = cap.read()

    size = frame.shape

    focal_length = size[1]
    center = (size[1]/2, size[0]/2)

    camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion

    # Take the blur to be more accurate
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Convert the color from BGR to HSV
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    # Mask the frame, meaning take colors from range l_b to u_b
    mask = cv2.inRange(hsv, l_b, u_b)

    # Combine the images using bitwise and operation
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # We do the erosion method to erase further noise in the background
    kernel = np.ones((5, 5), np.uint8)
    res = cv2.erode(res, kernel)

    # Contours is a Python list of ALL the contours in the image
    # This will be a numpy array in x,y values
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Approximates the contour into a shape with less verticies
        approx = cv2.approxPolyDP(cnt, 0.002 * cv2.arcLength(cnt, True), True)

        # Draws the approximated contour
        cv2.drawContours(res, [approx], 0, (0, 255, 0), 5)

        x = approx.ravel()[0]
        y = approx.ravel()[1]

        '''
        if area > 14:
            if 4 <= len(approx) <= 7:
                cv2.putText(res, "It is a Rectangle", (x, y), font, 1, (0, 255, 0))
            elif 8 <= len(approx) <= 10:
                cv2.putText(res, "Probably a circle", (x, y), font, 1, (0, 255, 0))
        # print("Area is " + str(area))
        '''

        # Coordinates of approximated contour's extremes
        # We take the contour and find the argument with the smallest/biggest
        # value and return the full coordinate
        leftmost = tuple(approx[approx[:,:,0].argmin()][0])
        rightmost = tuple(approx[approx[:,:,0].argmax()][0])
        topmost = tuple(approx[approx[:,:,1].argmin()][0])
        bottommost = tuple(approx[approx[:,:,1].argmax()][0])

        middleLine = (topmost[1] + bottommost[1]) / 2

        lowerCoordinates = []
        upperCoordinates = []

        for coor in approx:
            if coor[0][1] > middleLine:
                upperCoordinates.append([float(i) for i in coor[0]])
            else:
                lowerCoordinates.append([float(i) for i in coor[0]])

        topLeft = (99999,0)
        topRight = (0, 0)

        bottomLeft = (99999,0)
        bottomRight = (0, 0)

        for coordinate in upperCoordinates:
            if coordinate[0] < topLeft[0]:
                topLeft = tuple(coordinate)
            if coordinate[0] > topRight[0]:
                topRight = tuple(coordinate)

        for coordinate in lowerCoordinates:
            if coordinate[0] < bottomLeft[0]:
                bottomLeft = tuple(coordinate)
            if coordinate[0] > bottomRight[0]:
                bottomRight = tuple(coordinate)

        
        capturedTargetPoints = np.array([
                            topLeft,     # Top left corner
                            topRight,     # Top right corner
                            bottomLeft,     # Bottom left corner
                            bottomRight      # Bottom right corner
                            ])

        # print(capturedTargetPoints)

        # Draw rectangle around contour
        cv2.rectangle(res,(leftmost[0],topmost[1]),(rightmost[0],bottommost[1]),(0,255,0),3)

        if area > 500:
            (success, rotation_vector, translation_vector) = cv2.solvePnP(realTargetPoints, capturedTargetPoints, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_AP3P)
            R = cv2.Rodrigues(rotation_vector)[0]
            roll = 180*atan2(-R[2][1], R[2][2])/pi
            pitch = 180*asin(R[2][0])/pi
            yaw = 180*atan2(-R[1][0], R[0][0])/pi
            rot_params= [roll,pitch,yaw]
            print(rot_params)
            cv2.putText(res, str(rotation_vector), (leftmost[0], topmost[1]), font, 1, (0, 255, 0))

    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("res", res)

    # Check for keyboard input, 'escape' to quit
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        time.sleep(3)


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
