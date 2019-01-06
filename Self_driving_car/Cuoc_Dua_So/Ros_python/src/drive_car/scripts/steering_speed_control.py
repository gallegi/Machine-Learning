#!/usr/bin/env python
import thread
import time
import rospy
import cv2
from std_msgs.msg import Float32

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Lambda, Dropout

# ======================= NECESSARY IMPORT FOR SSD ============================
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, CSVLogger
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd7 import build_model
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast
#======================================================================================

# ======================= GLOBAL VALUE TO CONTROL CAR ============================
speed_amount = 10
steer_amount = 15
MIN_SPEED = 30
MAX_SPEED = 50
#======================================================================================

# ======================= THIS IS GLOBAL VARIABLES FOR SSD ============================
img_height = 480 # Height of the input images
img_width = 640 # Width of the input images
img_channels = 3 # Number of color channels of the input images
intensity_mean = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
intensity_range = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
n_classes = 3 # Number of positive classes
scales = [0.08, 0.16, 0.32, 0.64, 0.96] # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
aspect_ratios = [0.8, 1.0, 1.25] # The list of aspect ratios for the anchor boxes
two_boxes_for_ar1 = True # Whether or not you want to generate two anchor boxes for aspect ratio 1
steps = None # In case you'd like to set the step sizes for the anchor box grids manually; not recommended
offsets = None # In case you'd like to set the offsets for the anchor box grids manually; not recommended
clip_boxes = True # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [1.0, 1.0, 1.0, 1.0] # The list of variances by which the encoded target coordinates are scaled
normalize_coords = True # Whether or not the model is supposed to use coordinates relative to the image size
classes = ['neg','left','right','stop']
#======================================================================================

#============================= CAR BEHAVIOURAL CLONING CLASS ==========================
class Car:
    def __init__(self, listener, sign_detector):
        # some common variable to control the car
        self.speed = MAX_SPEED
        self.steer = 0
        self.key = None
        self.save_count = 0
        self.recording = False
        
        self.speed_pub = rospy.Publisher("Team1_speed",Float32,queue_size=10)
        self.steerAngle_pub = rospy.Publisher("Team1_steerAngle",Float32,queue_size=10)

        # image listener get current frame
        self.listener = listener

        # traffic sign detector to get current traffic sign
        self.sign_detector = sign_detector

        # define model
        self.model = Sequential()
        self.model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(240, 320,3)))
        self.model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
        self.model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
        self.model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
        self.model.add(Conv2D(64, 3, 3, activation='elu'))
        self.model.add(Conv2D(64, 3, 3, activation='elu'))
        self.model.add(Dropout(0.3))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='elu'))
        self.model.add(Dense(64, activation='elu'))
        self.model.add(Dense(32, activation='elu'))
        self.model.add(Dense(1))

        # load model's pretrained weight
        self.model.load_weights("models/my_model_01_04_14_05.h5")

        img = cv2.imread("/home/namntse05438/Cuoc_Dua_So/Ros_python/traffic_sign_data/1545984670.3_50_0.0.jpg")

        res = self.model.predict(img.reshape(1,240,320,3))
        print(res)


    def pub_speed_and_steer(self):
        while(True):
            img = None
            if(self.listener.has_image==False):
                continue
            else:
                img = self.listener.image_np.copy()
            cv2.imshow('color', img)
            cv2.waitKey(1)
            self.steer = float(self.model.predict(img.reshape(1,240,320,3)))
            #print(self.steer)

            if(self.sign_detector.sign == 1):
                self.steer -= 15
                print("Turn left")
            elif(self.sign_detector.sign == 2):
                self.steer += 15
                print("Turn right")

            self.speed_pub.publish(self.speed)
            self.steerAngle_pub.publish(self.steer)

            self.steer -= self.steer/4.0


    def manually_control(self):
        while(True):
            img = None
            self.save_count += 1
        
            if(self.key == ord('w')):
                #print("Up", speed_amount)
                self.speed += speed_amount
                if(self.speed > MAX_SPEED):
                    self.speed = MAX_SPEED
            elif(self.key==ord('s')):
                #print("Down", speed_amount)
                self.speed -= speed_amount
                if(self.speed<MIN_SPEED):
                    self.speed = MIN_SPEED
            elif(self.key==ord('a')):
                #print("Left", steer_amount)
                self.steer -= steer_amount
            elif(self.key==ord('d')):
                #print("Right", steer_amount)
                self.steer += steer_amount
            elif(self.key==ord('r')):
                self.steer = 0
                print("Reset steer")
            elif(self.key==ord('g')):
                self.recording = not self.recording
            elif(self.key == ord('f')):
                print("Exit by key")
                exit()

            if(self.listener.has_image==False):
                continue
            else:
                img = self.listener.image_np.copy()

            cv2.imshow('color', img)
            self.key = cv2.waitKey(8)

            if(self.save_count == 5):
                self.save_count = 0
                if(self.recording):
                    image_name = "./tmp/"+str(time.time())+"_"+str(self.speed)+"_"+str(self.steer)+".jpg"
                    cv2.imwrite(image_name, img)

            self.speed_pub.publish(self.speed)
            self.steerAngle_pub.publish(self.steer)
            self.steer -= self.steer/4.0
#======================================================================================


# ========================== SSD TRAFFIC SIGN DETECTOR CLASS ================================
class Traffic_Sign_Detector:
    def __init__(self, listener):

        # label of traffic sign detected by this traffic sign detector
        self.sign = -1

        # image lister object which can provide current frame
        self.listener = listener

        # build model
        self.model = build_model(image_size=(img_height, img_width, img_channels),
                            n_classes=n_classes,
                            mode='training',
                            l2_regularization=0.0005,
                            scales=scales,
                            aspect_ratios_global=aspect_ratios,
                            aspect_ratios_per_layer=None,
                            two_boxes_for_ar1=two_boxes_for_ar1,
                            steps=steps,
                            offsets=offsets,
                            clip_boxes=clip_boxes,
                            variances=variances,
                            normalize_coords=normalize_coords,
                            subtract_mean=intensity_mean,
                            divide_by_stddev=intensity_range)

        # 2: Optional: Load some weights
        self.model.load_weights('models/ssd7_epoch-04_loss-0.2610_val_loss-0.6570.h5', by_name=True)

        # 3: Instantiate an Adam optimizer and the SSD loss function and compile the model
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
        self.model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

        img = cv2.imread("/home/namntse05438/Cuoc_Dua_So/Ros_python/traffic_sign_data/1545920226.49_50_1.54542034912e-05.jpg")
        img = self.preprocessing(img)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        y_pred_decoded = self.get_ypred_decoded(rgb_img.reshape(1, img_height, img_width, 3))
        print(y_pred_decoded[0])
        for box in y_pred_decoded[0]:
            xmin = int(box[-4])
            ymin = int(box[-3])
            xmax = int(box[-2])
            ymax = int(box[-1])
            label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
            print(label)
            #draw bounding box
            cv2.rectangle(img, (xmin, ymin), (xmax,ymax), (0,0,255), 2)
            cv2.putText(img, str(label), (xmin,ymin+20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
        
        # cv2.imshow('traffic_sign_test', img)
        cv2.waitKey(1)

    # processing image to match the required input of ssd7 model
    def preprocessing(self, cv_img):
        return cv2.resize(cv_img, (img_width, img_height))

    # The following function to make prediction of ssd7 model
    def get_ypred_decoded(self, r_img):
        y_pred = self.model.predict(r_img)
        #y_pred = model.predict(r_img)
        y_pred_decoded = decode_detections(y_pred,
                                        confidence_thresh=0.9,
                                        iou_threshold=0.001,
                                        top_k=200,
                                        normalize_coords=normalize_coords,
                                        img_height=img_height,
                                        img_width=img_width)

        np.set_printoptions(precision=2, suppress=True, linewidth=90)
        
        return y_pred_decoded


    def set_traffic_sign(self):
        while(True):
            img = None
            self.sign = -1
            if(self.listener.has_image==False):
                continue
            else:
                # convert to rgb image, because ssd model was trained on rgb images, not bgr ones
                img = self.preprocessing(self.listener.image_np)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            r_img = img.reshape(1, img_height, img_width, 3)
            y_pred_decoded = self.get_ypred_decoded(r_img)
            
            for box in y_pred_decoded[0]:
                xmin = int(box[-4])
                ymin = int(box[-3])
                xmax = int(box[-2])
                ymax = int(box[-1])
                label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
                print(label)

                # set current traffic sign
                self.sign = int(box[0])

                #draw bounding box
                cv2.rectangle(img, (xmin, ymin), (xmax,ymax), (255,255,255), 2)
                cv2.putText(img, str(label), (xmax,ymax), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            cv2.imshow('traffic sign', img)
            cv2.waitKey(1)

# ===================================================================================================