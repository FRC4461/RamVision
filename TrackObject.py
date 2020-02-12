# Finished: Can track color with cam
# ToDo: Give unique ID? Publish network tables, connect from FRCVision to RoboRIO, Do contours for more accurate tracking

from __future__ import print_function
import numpy as np
import cv2


def nothing(x):
    pass


# HSV Notes
# H - Hue (Dominant Wavelength)
# S - Saturation (Purity / Shade of the color)
# V - Value (Intensity)

# ------Current "good" tracking settings that I found --------#
# Lower Hue = 69
# Lower Saturation = 121
# Lower Value = 151

# Upper Hue = 11
# Upper Saturation = 239
# Upper Value = 255

cap = cv2.VideoCapture(1)

# Create a frame that has sliders to calibrate value
cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)

while True:
    # Maybe I want to put this into a function as well (Would that slow down the program?)
    # Get the position of the lower limit of color
    l_h = cv2.getTrackbarPos("LH", "Tracking")
    l_s = cv2.getTrackbarPos("LS", "Tracking")
    l_v = cv2.getTrackbarPos("LV", "Tracking")

    # Get the position of the upper limit of color
    u_h = cv2.getTrackbarPos("UH", "Tracking")
    u_s = cv2.getTrackbarPos("US", "Tracking")
    u_v = cv2.getTrackbarPos("UV", "Tracking")

    # This will be dynamic for now for calibration purposes
    # Define the lower color range
    l_b = np.array([l_h, l_s, l_v])
    # Define the upper color range
    u_b = np.array([u_h, u_s, u_v])

    # Read the frame from the camera
    # Since I have it connected to computer it will be 1
    # Most likely for the Raspberry Pi it will be 0
    # We can have a fail safe so that for sure there is a camera stream
    _, frame = cap.read(1)
    # Take the blur to be more accurate
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Convert the color from BGR to HSV
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    # Mask the frame, meaning take colors from range l_b to u_b
    mask = cv2.inRange(hsv, l_b, u_b)

    # Combine the images using bitwise and operation
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # Contours is a Python list of ALL the contours in the image
    # This will be a numpy array in x,y values
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)

        print(area)

    cv2.drawContours(res, contours, -1, (0, 0, 255), 3)
    print(contours)
    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("res", res)

    # Check for keyboard input, 'escape' to quit
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
