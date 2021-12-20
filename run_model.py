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
#-----------------RUNNING THE MODEL----------------#

def video_detector():
    resultlist=[]
    timestamps=[]
    model = tf.keras.models.load_model(r"C:\\Users\\Zoe Mercury\\Desktop\\Iron II\\A.I\\Untitled Folder\\model_small.h5")
    face_cascade = cv2.CascadeClassifier("C:\\Users\\Zoe Mercury\\Desktop\\Iron II\\A.I\\Untitled Folder\\cropped\\haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)

    emotions = ('sad','happy','angry','excited')

    while True:
        ret, img = cap.read()
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray=frame[y:y+w,x:x+h]
            roi_gray=cv2.resize(frame,(200,200))
            img_pixels = np.array(roi_gray)  / 255
            img_pixels = np.expand_dims(img_pixels, axis = 0)

            predictions = model.predict(img_pixels)
            print(predictions)
            max_index = np.argmax(predictions[0])
            print(max_index)
            resultlist.append(max_index)
            timestamps.append((datetime.now()))
            d= {'Video Prediction': resultlist,'Time':timestamps}
            df = pd.DataFrame(d, columns=['Video Prediction','Time'])

            predicted_emotion = emotions[max_index] + ' '+ str(int(predictions[0][max_index]*100)) +'%'

            cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


        cv2.imshow('img',img)

        if cv2.waitKey(10) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return df

#--------------COMPARE RESULTS----------------

df.merge(comparison,on='Time')

if df['EEG Prediction']==df['Video Prediction']:
    print('You are honest!')
else:
    print('Mismatch of Emotion and Facial Expression!')