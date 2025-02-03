import cv2
import numpy
from matplotlib import pyplot


def simplify_frame(frame): 
    pass

def main(): 
    cam = cv2.VideoCapture(0)
    frame_number = 0 
    frame_captured = True
    while True:
        frame_captured, frame = cam.read() 
        if not frame_captured: 
            break 
        frame_number += 1 
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        filename = 'img{:d}.jpg'.format(frame_number) 
        cv2.imwrite(filename = filename, img = frame) 

if __name__ == "__main__": 
    main() 


"""
This function....(purpose of the function/what it does)

Parameters: 
a (data type): YYY.
a (data type): YYY.

Returns: 
(data type): YYY. 

"""