#!/usr/bin/env python
from __future__ import print_function
import rospy
from std_msgs.msg import Float32
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

# import custom library to read image, lane detect and traffic sign detect
from image_listener import *
from steering_speed_control import *

manual = False

def main(args):
    # init node to communicate with simulator, could be any name
    rospy.init_node('car_driver', anonymous=True)

    # listener object to get image streamed back from simulator
    listener = ImageListener()

    # traffic sign detector object to detect current traffic sign
    sign_detector = Traffic_Sign_Detector(listener)

    # car object to control the car (speed, steer)
    car = Car(listener, sign_detector)

    # check if drive manually
    if(manual):
        # get key w a s d to drive car from key board
        thread.start_new_thread(car.manually_control(), ())
    else:
        # auto driving
        # start thread to detect traffic sign
        thread.start_new_thread(sign_detector.set_traffic_sign, ())

        # start thread to control the car
        thread.start_new_thread(car.pub_speed_and_steer, ())

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
