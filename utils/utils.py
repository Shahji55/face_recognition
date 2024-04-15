import math

import cv2
import numpy as np
from PIL import Image

def letterbox_image(image, size):
    ih, iw, _   = np.shape(image)
    w, h        = size
    scale       = min(w/iw, h/ih)
    nw          = int(iw*scale)
    nh          = int(ih*scale)

    image       = cv2.resize(image, (nw, nh))
    new_image   = np.ones([size[1], size[0], 3]) * 128
    new_image[(h-nh)//2:nh+(h-nh)//2, (w-nw)//2:nw+(w-nw)//2] = image
    return new_image
    
def preprocess_input(image):
    image -= np.array((104, 117, 123),np.float32)
    return image

def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    # (n, )
    return np.linalg.norm(face_encodings - face_to_compare, axis=1)

def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=1):
    dis = face_distance(known_face_encodings, face_encoding_to_check) 
    return list(dis <= tolerance), dis

def Alignment_1(img,landmark):
    if landmark.shape[0]==68:
        x = landmark[36,0] - landmark[45,0]
        y = landmark[36,1] - landmark[45,1]
    elif landmark.shape[0]==5:
        x = landmark[0,0] - landmark[1,0]
        y = landmark[0,1] - landmark[1,1]

    if x==0:
        angle = 0
    else: 
        angle = math.atan(y/x)*180/math.pi

    center = (img.shape[1]//2, img.shape[0]//2)
    
    RotationMatrix = cv2.getRotationMatrix2D(center, angle, 1)

    new_img = cv2.warpAffine(img,RotationMatrix,(img.shape[1],img.shape[0])) 

    RotationMatrix = np.array(RotationMatrix)
    new_landmark = []
    for i in range(landmark.shape[0]):
        pts = []    
        pts.append(RotationMatrix[0,0]*landmark[i,0]+RotationMatrix[0,1]*landmark[i,1]+RotationMatrix[0,2])
        pts.append(RotationMatrix[1,0]*landmark[i,0]+RotationMatrix[1,1]*landmark[i,1]+RotationMatrix[1,2])
        new_landmark.append(pts)

    new_landmark = np.array(new_landmark)

    return new_img, new_landmark
