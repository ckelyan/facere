import os
import cv2 as cv
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, 'images')

frontal_face_cascade = cv.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
recognizer = cv.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith('png') or file.endswith('jpg'):
            path = os.path.join(root, file)
            label = os.path.basename(root)
            
            if label in label_ids.keys():
                pass
            else:
                label_ids[label] = current_id
                current_id += 1
                
            id_ = label_ids[label]
            
            pil_image = Image.open(path).convert('L')
            image_array = np.array(pil_image, 'uint8')
            
            faces = frontal_face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
            
            for x, y, w, h in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
                
with open('trained_data/labels.pkl', 'wb') as f:
    pickle.dump(label_ids, f)
    
recognizer.train(x_train, np.array(y_labels))
recognizer.save('trained_data/trainer.yml')