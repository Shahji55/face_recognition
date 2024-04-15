import os
from retinaface import Retinaface
import cv2
import numpy as np

class Driver():

    def __init__(self):
        self.retinaface = Retinaface()

    # Face detection and its features extraction
    def encode(self):

        faces_path = "img/"
        list_dir = os.listdir(faces_path)

        image_paths = []

        names = []

        for name in list_dir:

            image_paths.append(faces_path + name)
            names.append(name.split(".")[0])

        print("Names: ", names)

        self.retinaface.encode_face_dataset(image_paths,names)

    # Face detection, recognition and output visualization
    def predict(self, img_path):
        
        img = cv2.imread(img_path)

        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
        #frame = np.array(self.retinaface_predict.detect_image(frame))
        img, boxes_conf_landms, name = self.retinaface.detect_image(img)

        #print("Here: ", boxes_conf_landms)

        if boxes_conf_landms is not None and name is not None:
            #print(boxes_conf_landms)
            dets = []

            for i, b in enumerate(boxes_conf_landms):
                #b = list(map(int, b))
                #print(b)

                x1 = int(b[0])
                y1 = int(b[1])
                x2 = int(b[2])
                y2 = int(b[3])

                conf = b[4]

                dets.append([x1, y1, x2, y2, conf])

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, name, (x1, y1 - 2), 0, 2 / 3, (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    
        img = np.array(img)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        
        cv2.imwrite("output.jpg", img)

if __name__ == "__main__":
    obj = Driver()
    obj.encode()

    #img_path = "img/real_id.jpg"
    #obj.predict(img_path)
