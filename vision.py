import numpy as np
import cv2 as cv
import pickle

class Vision:
    def __init__(self):
        frontal_face_cascade = cv.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
        # profile_face_cascade = cv.CascadeClassifier('cascades/data/haarcascade_profileface.xml')
        self.cascade = frontal_face_cascade
        
        self.recognizer = cv.face.LBPHFaceRecognizer_create()
        self.recognizer.read('trained_data/trainer.yml')
        
        with open('trained_data/labels.pkl', 'rb') as f:
            self.labels = pickle.load(f)
            
        self.labels = {v:k for k, v in self.labels.items()}
        print(self.labels)
    
    def find_faces(self, frame, still=False):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

        for x, y, w, h in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            id_, conf = self.recognizer.predict(roi_gray)

            recognized_face = 'none'
            if conf <= 120:
                recognized_face = self.labels[id_]
                print('Recognized', recognized_face)

            top_left = (x, y)
            bottom_right = (x + w, y + h)
            cv.rectangle(frame, top_left, bottom_right, color=(0, 0, 255), lineType=cv.LINE_4, thickness=4)
            
            text = f'{recognized_face}, {int(conf)}'
            font = cv.FONT_HERSHEY_COMPLEX
            font_scale = 2
            font_thickness = 2
            textpos = (x, y + h+50)
            tx, ty = textpos
            
            text_size, _ = cv.getTextSize(text, font, font_scale, font_thickness)
            text_w, text_h = text_size
            
            cv.rectangle(frame, textpos, (tx + text_w, ty - text_h), (0, 0, 0), -1)
            cv.putText(frame, text, textpos, fontFace=font, color=(0, 0, 255), fontScale=font_scale, thickness=font_thickness, lineType=cv.LINE_AA)
            
        cv.imshow('Faces', frame)
        
        if still:
            cv.waitKey(0)