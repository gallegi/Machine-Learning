#!/usr/bin/env python
from __future__ import print_function
import rospy

import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
import sys
import os
import time
import pandas as pd
import thread

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Lambda, Dropout

speed_amount = 10
steer_amount = 10
MIN_SPEED = 30
MAX_SPEED = 60

class ImageListener:

    def __init__(self):
        # control if not receive images yet, not use model to get steer
        self.has_image = False
        self.image_np = None
        self.key = None
        self.subscriber = rospy.Subscriber("Team1_image/compressed",CompressedImage, self.callback, queue_size=1)

    def callback(self, ros_data):

        np_arr = np.fromstring(ros_data.data, np.uint8)
        self.image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.has_image = True
        
        #image_name = "./image_data/"+str(time.time())+"_"+str(self.speed)+"_"+str(self.steer)+".jpg"
        #cv2.imwrite(image_name, image_np)
        # cv2.imshow("color", self.image_np)
        # self.key = cv2.waitKey(1)