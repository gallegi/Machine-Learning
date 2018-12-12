import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from time import time
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.svm import SVC
from sklearn.externals import joblib

path_to_model = "tf_sign.pkl"

classes = ['00017','00033', '00034']


W = 28
H = 28
std_confnd = 0.6

def preprocessing_img(img):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  ret = cv2.resize(gray,(W,H))
  return ret

def load_svm_model():
    clf = joblib.load(path_to_model)
    return clf

def test_model(clf):
    # testing phase
    _X = []
    _y = []

    for i in range(len(classes)):
        _class = classes[i]
        path = '/home/namntse05438/datasets/GTSRB/mydataval/' + _class + '/'
        image_names = glob.glob(path + '*.ppm')

        for img_name in image_names:
            image = cv2.imread(img_name, 0)
            image = preprocessing_img(image)
            image = np.reshape(image, W * H)
            _X.append(image)
            _y.append(i)

    _X = np.array(_X).astype('float32')
    _y = np.array(_y)

    # predict on test set
    print("Predicting tf_sign on the test set")
    t0 = time()
    _y_pred = clf.predict(_X)
    print("done in %0.3fs" % (time() - t0))

    print(classification_report(_y, _y_pred, target_names=classes))
    print(confusion_matrix(_y, _y_pred, labels=range(len(classes))))


class Traffic_sign_recognition():
    candidates = []
    image = None

    def __init__(self, image, clf):
        self.image = image
        self.clf = clf

    def blue_detector(self, input):
        ret = cv2.inRange(input, (100,100,70), (120,255,255))
        return ret

    def extract_cropped_images(self, mask):
        candidates = []

        #find contour with privided mask
        im2, cnts, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in cnts:
            x,y,w,h = cv2.boundingRect(cnt)

            area = w*h

            if(area < 20):
                continue

            cropped = self.image[y:y+w+1,x:x+h+1]
            self.candidates.append(cropped)
            #cv2.imshow("cropped", cropped)



    # extract candidate
    def extract_candidate(self):
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        blue_mask = self.blue_detector(hsv)
        cv2.imshow("mask", blue_mask)
        self.extract_cropped_images(mask=blue_mask)


    def get_label(self, cropped_img):
        reshaped_img = np.reshape(cropped_img, (1,-1))   #flatten the image to a row data

        prob = y_pred = self.clf.predict_proba(reshaped_img)
        max_idx = np.argmax(prob)
        if(prob[0,max_idx] >= std_confnd):
            return max_idx, prob[0,max_idx]

        return -1, -1

    def test(self):
        self.extract_candidate()
        for cropped in self.candidates:
            #resize cropped to fit to svm model
            cropped = preprocessing_img(cropped)
            print(cropped.shape)
            label, prob = self.get_label(cropped)
            print(label," ", prob)


def main():
    clf = load_svm_model()

    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):
        ret, frame = cap.read()
        frame = cv2.flip(frame,1)

        recognition = Traffic_sign_recognition(frame, clf)
        recognition.test()

        key = cv2.waitKey(30)
        if(key == 102):
            return
        elif(key == 115):
            cv2.waitKey(0)




if __name__ == '__main__':
    main()