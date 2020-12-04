import torch
import numpy as np


from sklearn import model_selection
from sklearn import metrics

from config import *
import dataset

def training():
    train_img, test_img, train_labels, test_labels, train_orig_labels, test_orig_targets = model_selection.train_test_split(IMAGES, LABELS_ENCODED, LABELS_NAMES, test_size=0.1, random_state=2020)
    
    
    
  


