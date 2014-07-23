import cv2
import cv
import time
from facedetect import *


if __name__ == '__main__':
    
    video_file_name = '../Test_Tracking_Movie.mov'
    cap = cv2.VideoCapture(video_file_name)
    while(cap.isOpened()):
        ret, frame = cap.read()
    
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        cv2.imshow('frame',frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            cap.release()
            break