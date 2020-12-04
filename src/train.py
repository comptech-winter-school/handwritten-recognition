import os
import glob
import torch
import numpy as np

from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics

from config import *
import dataset

def training():
    images = glob.glob(os.path.join(DATASET_PATH, ".png"))
    labels_names = [x.split('/') for x in images]

images = glob.glob(os.path.join(DATASET_PATH, ".png"))
labels_names = [x.split('/')[-1][:-4] for x in images]
