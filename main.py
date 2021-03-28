import numpy as np
from PIL import Image
from pathlib import Path
from feature_extra import cal_feature
from model import SVM_model
from sklearn.svm import SVC
import matplotlib.pyplot as plt

Dataset = './bio/dataset/training/'
Dataset_test = './bio/dataset/testing/'

train = []
for i, _dir in enumerate(Path(Dataset).glob('*')):
    train.append(_dir)
test = []
for i, _dir in enumerate(Path(Dataset_test).glob('*')):
    test.append(_dir)

label = [1 , 0 , 0 , 1]
        #[Benign,Malignant,Malignant,Benign]
# train dataset
ROI = [[500,650,580,780],[630,770,270,400],[220,880,120,720],[390,750,400,729]]
train_x = []
for i,pat in enumerate(train):
    I = Image.open(pat).convert('L')
    img = np.asarray(I)
    img_roi = img[ ROI[i][0]:ROI[i][1] , ROI[i][2]:ROI[i][3] ]
    ribbon_length = 4
    F = cal_feature(img_roi,ribbon_length)
    train_x.append([])
    features = F.main()
    train_x[i]=features

# test dataset
ROI = [[430,900,350,650]]
test_x = []
for i,pat in enumerate(test):
    I = Image.open(pat).convert('L')
    img = np.asarray(I)
    img_roi = img[ROI[i][0]:ROI[i][1], ROI[i][2]:ROI[i][3]]
    ribbon_length = 2
    F = cal_feature(img_roi, ribbon_length)
    test_x.append([])
    features = F.main()
    test_x[i]=features


s = SVM_model(train_x,label,test_x)
s.train()
print(s.test())









