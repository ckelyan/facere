from vision import Vision
import cv2 as cv

frame = cv.imread('obamaface.jpg')

vi = Vision()

vi.find_faces(frame, True)