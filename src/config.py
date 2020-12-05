import glob
import os
from sklearn import preprocessing
import numpy as np


DATASET_PATH = "/content/captcha_images_v2"
BATCH_SIZE = 32
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 75
NUM_WORKERS = 4
EPOCHS = 200
DEVICE = 'cuda'


IMAGES = glob.glob(os.path.join(DATASET_PATH, "*.png"))
# to look like '6bnnm'
LABELS_NAMES = [x.split('/')[-1][:-4] for x in IMAGES]
# to look like ['g', 'p', 'x', 'n', 'g']
LABELS_NAMES = [[_ for _ in x] for x in LABELS_NAMES]
LABELS_NAMES_FLAT = [_ for sublist in LABELS_NAMES for _ in sublist]
labels_encoded = preprocessing.LabelEncoder()
labels_encoded.fit(LABELS_NAMES_FLAT)
# print(labels_encoded.classes_)
# keep 0 for unknown
LABELS_ENCODED = np.array([labels_encoded.transform(x) for x in LABELS_NAMES]) +1
# print(LABELS_ENCODED)
# print(len(labels_encoded.classes_))