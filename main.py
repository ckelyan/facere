import numpy as np
import cv2 as cv
from vision import Vision

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print('Coulnd\'t get a video stream from a camera, exiting...')
    exit()

vi = Vision()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print('Did not receive current frame, exiting...')
        exit()
    
    vi.find_faces(frame)    
    
    # cv.imshow('Video stream', frame)
    
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()