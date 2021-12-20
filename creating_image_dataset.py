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

# -----------Unpacking fer2013 image library----------#

def create_image(column):
    images = pd.read_csv('fer2013.csv')
    width = 48
    height = 48
    a = 0
    b = 0
    c = 0
    d = 0
    e = 0
    f = 0
    g = 0
    img = Image.new(mode="RGB", size=(width, height))
    img.putdata(pixels)
    if images['emotion'] == 0:
        img.save('angry_' + str(a) + '.png')
        a += 1
    elif images['emotion'] == 1:
        img.save('disgust_' + str(b) + '.png')
        b += 1
    elif images['emotion'] == 2:
        img.save('fear_' + str(c) + '.png')
        c += 1
    elif images['emotion'] == 3:
        img.save('happy_' + str(d) + '.png')
        d += 1
    elif images['emotion'] == 4:
        img.save('sad_' + str(e) + '.png')
        e += 1
    elif images['emotion'] == 5:
        img.save('surprise_' + str(f) + '.png')
        f += 1
    else:
        img.save('neutral_' + str(g) + '.png')
        g += 1


images['pixels'].apply(create_image)

# 0: -4593 images- Angry
# 1: -547 images- Disgust
# 2: -5121 images- Fear
# 3: -8989 images- Happy
# 4: -6077 images- Sad
# 5: -4002 images- Surprise


