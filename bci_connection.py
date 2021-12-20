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

# INSTALLING BCI CONNECTION

#--------------- SETUP--------------#

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
  name = 'pyOpenBCI',
  packages = find_packages(),
  version = '0.13',
  license='MIT',
  description = 'A lib for controlling OpenBCI devices',
  long_description=long_description,
  long_description_content_type="text/markdown",
  author = 'OpenBCI, Inc.',
  author_email = 'contact@openbci.com',
  url = 'https://github.com/andreaortuno/pyOpenBCI',
  download_url = 'https://github.com/andreaortuno/pyOpenBCI/archive/0.13.tar.gz',
  keywords = ['device', 'control', 'eeg', 'emg', 'ekg', 'ads1299', 'openbci', 'ganglion', 'cyton', 'wifi'],
  install_requires=[
          'numpy',
          'pyserial',
          'bitstring',
          'xmltodict',
          'requests',
      ] + ["bluepy >= 1.2"] if sys.platform.startswith("linux") else [],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.6',
  ],
)

# https://github.com/openbci-archive/pyOpenBCI/blob/master/setup.py

# START STREAM

def print_raw(sample):
    print(sample.channels_data)

board = OpenBCICyton()
#Set (daisy = True) to stream 16 ch
board = OpenBCICyton(daisy = False)

board.start_stream(print_raw)
# turn into dataframe and add timestamp column 'Time'
