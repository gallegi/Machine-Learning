import cv2
import glob
import numpy as np

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

class Data_Loader(object):

    image_data = []
    label_data = []

    def __init__(self, path_to_folder, classes,img_w=32, img_h=32, test_size=0.3):
        self.path_to_folder = path_to_folder
        self.classes = classes
        self.img_w = img_w
        self.img_h = img_h
        self.test_size = test_size

    # set content to self.image_date and self.label_data
    def read_data(self):
        print('Reading data ...')
        for clss in self.classes:
            class_index = self.classes.index(clss)
            print("Reading class "+clss +" - index "+str(class_index))
            image_names = glob.glob(self.path_to_folder + '/' + clss + '/*.ppm')
            for img_name in image_names:
                img = cv2.imread(img_name)
                if (img is None):
                    print('cannot read images '+image_names)
                    SystemExit(0)

                processed_img = self.preprocessing_image(img)
                self.image_data.append(processed_img)
                self.label_data.append(class_index)
                # cv2.imshow("image", processed_img)
                # cv2.waitKey(0)
        print('Read data successfully !!!')


    def preprocessing_image(self, img):
        resize_img = cv2.resize(img, (self.img_w, self.img_h))
        return resize_img


    def getTrainTestSets(self):
        self.read_data()

        # convert data to numpy array
        self.image_data = np.array(self.image_data).astype(np.float32)
        self.label_data = np.array(self.label_data).astype(np.int64)

        # scale data mean = 0 and stddev = 1
        self.image_data = (self.image_data - 128) / 255.0
        # convert label to 1 hot vector
        self.label_data = to_categorical(self.label_data)

        # shuffle data_set
        train_images, test_images, train_labels, test_labels = train_test_split(self.image_data, self.label_data,
                                                                                test_size=self.test_size)

        return train_images, train_labels, test_images, test_labels