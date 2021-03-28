import numpy as np
from PIL import Image
from pathlib import Path
from feature_extra import cal_feature
from sklearn.svm import SVC
import matplotlib.pyplot as plt

class SVM_model():
    def __init__(self,train_x,train_y,test_x):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x

    def train(self):
        self.svm = SVC(kernel='linear', probability=True)
        self.svm.fit(self.train_x, self.train_y)

    def test(self):

        return self.svm.predict(self.test_x)
