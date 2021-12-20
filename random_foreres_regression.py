# IMPORTS
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import cv2
import sys
import tensorflow as tf
import keras
import time

from PIL import Image

from distutils.core import setup
from setuptools import find_packages

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense,BatchNormalization

import multiprocessing

#--------------------------------------------------------------------------------------------------------------

# PREDICTING EMOTION FROM EEG DATA

# Loading the csv data
eeg= pd.read_csv(r"C:\Users\Zoe Mercury\Desktop\Iron II\A.I\EEG emotions.csv" )

# Encoding the categories into numbers
# Neg= 0  Neu= 1 Pos= 2
le = preprocessing.LabelEncoder()
le.fit(eeg['label'])
eeg['labels']=le.transform(eeg['label'])
eeg2=eeg.drop('label',axis=1)

# Train Test Split
X=eeg2.drop('labels',axis=1)
y=eeg2['labels']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1)


# Random Forrest Regression
forest = RandomForestRegressor(n_estimators=12,
                               max_depth=5,
                               random_state=1)
forest.fit(X_train, y_train)
print('train accuracy was',forest.score(X_train,y_train))

pred = forest.predict(X_test)
print('test accuracy was',forest.score(X_test,y_test))
comparison=pd.DataFrame({'observed':y_test, 'predicted':pred})
display(comparison)


# Save model
pickle.dump(forest, open('C:\\Users\\Zoe Mercury\\Desktop\\Iron II\\A.I\\forest.pickle', 'wb'))
# PermissionError: [Errno 13] Permission denied: 'C:\\Users\\Zoe Mercury\\Desktop\\Iron II\\A.I' because I didn't add a filename+extension to the path

# Load model
forest_model = pickle.load(open('C:\\Users\\Zoe Mercury\\Desktop\\Iron II\\A.I\\forest.pickle','rb'))

# Model is receiving data from OpenBCI and predicting
# 1 brainwave input per second
# get BCI output and convert into dataframe
# figure out how to add a new row every instance and only predict for the new row
# then pred= forest.predict(df)

