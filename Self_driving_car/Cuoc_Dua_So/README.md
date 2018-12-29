# Car behavioral cloning:

This code use supervised learning with CNN to perform regression, which predicts the steering angle of the car. Speed is set contantly to 50. 
There are 3 thread in the program:
- main thread: listen to the images streamed by simulator
- second thread: use CNN to predict steering angle 
- third thread: use SSD to detect traffic sign (left, right, even stop :))
SSD still runs parallelly with the others, detect traffic sign but does not do anythings affect the steering. Because the second thread (predict steering angle) can learn the whole map, no need traffic sign detector

## How to run project:

### Main requirements:
- Ros python
- Tensorflow
- Keras
- Pandas (for train)
- Opencv-python

### Run project:

Ros_python/ is the folder contain code to drive car
Open a terminal,
```bash
cd Ros_python
catkin_make
source devel/setup.bash
roslaunch drive_car drive_car.launch
```
Open another one,
```bash
cd Ros_python
source devel/setup.bash
rosrun drive_car car_driver.py
```

### How to train on new data:
Use folder Train_lane/
```bash
cd Train_lane
jupyter notebook Train_Lane.ipynb
```
Please edit path to your data folder and NOTICE: the name of each image in this folder must be the following format:
Use folder Train_lane/
```bash
time_speed_steering.jpg
```
