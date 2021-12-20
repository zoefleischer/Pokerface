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

# PARALLELIZATION OF PROCESSES

start = time.perf_counter()
p1 = multiprocessing.Process(target = video_detector, args = [2])
p2 = multiprocessing.Process(target = eeg_detector, args = [2])

p1.start()
p2.start()

p1.join()
p2.join()

finish = time.perf_counter()
print('finished in ' + str(finish-start)+' seconds')