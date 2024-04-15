import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from nets.facenet import Facenet
from nets_retinaface.retinaface import RetinaFace
from utils.anchors import Anchors
from utils.config import cfg_mnet, cfg_re50
from utils.utils import (Alignment_1, compare_faces, letterbox_image,
                         preprocess_input)
from utils.utils_bbox import (decode, decode_landm, non_max_suppression,
                              retinaface_correct_boxes)


class Retinaface(object):
    _defaults = {

        "retinaface_model_path" : 'model_data/Retinaface_mobilenet0.25.pth',

        "retinaface_backbone"   : "mobilenet",

        "confidence"            : 0.5,

        "nms_iou"               : 0.3,

        "retinaface_input_shape": [640, 640, 3],

        "letterbox_image"       : True,
        
        "facenet_model_path"    : 'model_data/facenet_inception_resnetv1.pth',
        #"facenet_model_path"    : 'model_data/facenet_mobilenet.pth',

        "facenet_backbone"      : "inception_resnetv1",
        #"facenet_backbone"      : "mobilenet",

        "facenet_input_shape"   : [160, 160, 3],

        "facenet_threhold"      : 0.9,

        "cuda"                  : True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, encoding=0, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        if self.retinaface_backbone == "mobilenet":
            self.cfg = cfg_mnet
        else:
            self.cfg = cfg_re50

        self.anchors = Anchors(self.cfg, image_size=(self.retinaface_input_shape[0], self.retinaface_input_shape[1])).get_anchors()
        self.generate()

        try:
            print("Backbone: ", self.facenet_backbone)
            self.known_face_encodings = np.load("encoding_data/{backbone}_face_encoding.npy".format(backbone=self.facenet_backbone))
            self.known_face_names     = np.load("encoding_data/{backbone}_names.npy".format(backbone=self.facenet_backbone))
        except:
            # if not encoding:
            #     print("exception")
            pass

    def generate(self):

        self.net        = RetinaFace(cfg=self.cfg, phase='eval', pre_train=False).eval()
        self.facenet    = Facenet(backbone=self.facenet_backbone, mode="predict").eval()

        print('Loading weights into state dict...')
        state_dict = torch.load(self.retinaface_model_path)
        self.net.load_state_dict(state_dict)

        state_dict = torch.load(self.facenet_model_path)
        self.facenet.load_state_dict(state_dict, strict=False)

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

            self.facenet = nn.DataParallel(self.facenet)
            self.facenet = self.facenet.cuda()
        print('Finished!')

    # Extract face embeddings and save them
    def encode_face_dataset(self, image_paths, names):
        face_encodings = []
        for  index , path  in  enumerate ( tqdm ( image_paths )):
            #---------------------------------------------------#
            # Open face image
            #---------------------------------------------------#
            image       = np.array(Image.open(path), np.float32)
            #---------------------------------------------------#
            # make a backup of the input image
            #---------------------------------------------------#
            old_image   = image.copy()
            #---------------------------------------------------#
            # Calculate the height and width of the input image
            #---------------------------------------------------#
            im_height, im_width, _ = np.shape(image)
            #---------------------------------------------------#
            # Calculate scale, which is used to convert the obtained prediction frame into the height and width of the original image
            #---------------------------------------------------#
            scale = [
                np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]
            ]
            scale_for_landmarks = [
                np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                np.shape(image)[1], np.shape(image)[0]
            ]
            if self.letterbox_image:
                image = letterbox_image(image, [self.retinaface_input_shape[1], self.retinaface_input_shape[0]])
                anchors = self.anchors
            else:
                anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()

            #---------------------------------------------------#
            # Pass the processed image into the Retinaface network for prediction
            #---------------------------------------------------#
            with torch.no_grad():
                #-----------------------------------------------------------#
                # Image preprocessing, normalization.
                #-----------------------------------------------------------#
                image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(torch.FloatTensor)

                image = image.cuda()
                anchors = anchors.cuda()

                loc, conf, landms = self.net(image)
                #-----------------------------------------------------------#
                # Decode the prediction box
                #-----------------------------------------------------------#
                boxes   = decode(loc.data.squeeze(0), anchors, self.cfg['variance'])
                #-----------------------------------------------------------#
                # Get the confidence of the prediction result
                #-----------------------------------------------------------#
                conf    = conf.data.squeeze(0)[:, 1:2]
                #-----------------------------------------------------------#
                # Decode face key points
                #-----------------------------------------------------------#
                landms  = decode_landm(landms.data.squeeze(0), anchors, self.cfg['variance'])

                #-----------------------------------------------------------#
                # stack face detection results
                #-----------------------------------------------------------#
                boxes_conf_landms = torch.cat([boxes, conf, landms], -1)

                boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)
               
                if len(boxes_conf_landms) <= 0:
                    print ( names [ index ], ": no face detected" )
                    continue
                #---------------------------------------------------------#
                # If letterbox_image is used, remove the gray bar.
                #---------------------------------------------------------#
                if self.letterbox_image:
                    boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, \
                        np.array([self.retinaface_input_shape[0], self.retinaface_input_shape[1]]), np.array([im_height, im_width]))

            boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
            boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks

            #---------------------------------------------------#
            # Select the largest face frame.
            #---------------------------------------------------#
            best_face_location  = None
            biggest_area        = 0
            for result in boxes_conf_landms:
                left, top, right, bottom = result[0:4]

                w = right - left
                h = bottom - top
                if w * h > biggest_area:
                    biggest_area = w * h
                    best_face_location = result

            #---------------------------------------------------#
            # capture image
            #---------------------------------------------------#
            crop_img = old_image[int(best_face_location[1]):int(best_face_location[3]), int(best_face_location[0]):int(best_face_location[2])]
            #cv2.imwrite("face.jpg", crop_img)

            landmark = np.reshape(best_face_location[5:],(5,2)) - np.array([int(best_face_location[0]),int(best_face_location[1])])
            crop_img,_ = Alignment_1(crop_img,landmark)

            crop_img = np.array(letterbox_image(np.uint8(crop_img),(self.facenet_input_shape[1],self.facenet_input_shape[0])))/255
            crop_img = crop_img.transpose(2, 0, 1)
            crop_img = np.expand_dims(crop_img,0)
            #---------------------------------------------------#
            # Use the image to calculate the feature vector of length 128
            #---------------------------------------------------#
            with torch.no_grad():
                crop_img = torch.from_numpy(crop_img).type(torch.FloatTensor)
                crop_img = crop_img.cuda()

                face_encoding = self.facenet(crop_img)[0].cpu().numpy()
                face_encodings.append(face_encoding)

        np.save("encoding_data/{backbone}_face_encoding.npy".format(backbone=self.facenet_backbone),face_encodings)
        np.save("encoding_data/{backbone}_names.npy".format(backbone=self.facenet_backbone),names)

        #return face_encodings

    # Detect face, extract its features, compare the features against ground truth and make prediction
    def detect_image(self, image):

        old_image   = image.copy()

        image       = np.array(image, np.float32)

        im_height, im_width, _ = np.shape(image)

        scale = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]
        ]
        scale_for_landmarks = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0]
        ]

        if self.letterbox_image:
            image = letterbox_image(image, [self.retinaface_input_shape[1], self.retinaface_input_shape[0]])
            anchors = self.anchors
        else:
            anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()

        with torch.no_grad():

            image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(torch.FloatTensor)

            if self.cuda:
                anchors = anchors.cuda()
                image   = image.cuda()

            loc, conf, landms = self.net(image)

            boxes   = decode(loc.data.squeeze(0), anchors, self.cfg['variance'])

            conf    = conf.data.squeeze(0)[:, 1:2]

            landms  = decode_landm(landms.data.squeeze(0), anchors, self.cfg['variance'])

            boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
            
            # boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)
            boxes_conf_landms = non_max_suppression(boxes_conf_landms, conf_thres=0.75, nms_thres=0.35)
        
            if len(boxes_conf_landms) <= 0:
                return old_image, None, None

            if self.letterbox_image:
                boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, \
                    np.array([self.retinaface_input_shape[0], self.retinaface_input_shape[1]]), np.array([im_height, im_width]))

            boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
            boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks


        face_encodings = []
        for boxes_conf_landm in boxes_conf_landms:

            boxes_conf_landm    = np.maximum(boxes_conf_landm, 0)
            crop_img            = np.array(old_image)[int(boxes_conf_landm[1]):int(boxes_conf_landm[3]), int(boxes_conf_landm[0]):int(boxes_conf_landm[2])]
            landmark            = np.reshape(boxes_conf_landm[5:],(5,2)) - np.array([int(boxes_conf_landm[0]),int(boxes_conf_landm[1])])
            crop_img, _         = Alignment_1(crop_img, landmark)

            crop_img = np.array(letterbox_image(np.uint8(crop_img),(self.facenet_input_shape[1],self.facenet_input_shape[0])))/255
            crop_img = np.expand_dims(crop_img.transpose(2, 0, 1),0)
            with torch.no_grad():
                crop_img = torch.from_numpy(crop_img).type(torch.FloatTensor)
                if self.cuda:
                    crop_img = crop_img.cuda()

                face_encoding = self.facenet(crop_img)[0].cpu().numpy()
            
                face_encodings.append(face_encoding)

        face_names = []
        for face_encoding in face_encodings:

            matches, face_distances = compare_faces(self.known_face_encodings, face_encoding, tolerance = self.facenet_threhold)
            name = "Unknown" # If face does not match then prediction is "Unknown"

            #print("Face distance: ", face_distances, matches)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]: 
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        return old_image, boxes_conf_landms, name
