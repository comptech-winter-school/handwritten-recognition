import torch
import numpy as np


from sklearn import model_selection
from sklearn import metrics

from config import *
from dataset import OcrDataset


def dataloader():
    (
        train_img,
        test_img, 
        train_labels, 
        test_labels, 
        train_orig_labels, 
        test_orig_targets,
    )   =model_selection.train_test_split(
            IMAGES, LABELS_ENCODED, LABELS_NAMES, test_size=0.1, random_state=2020)
    
    train_dataset = OcrDataset(image_path=train_img, 
                               labels=train_labels, 
                               resize=(IMAGE_HEIGHT,IMAGE_WIDTH)
                              )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True
    )
    
    
    test_dataset = OcrDataset(image_path=test_img, 
                               labels=test_labels, 
                               resize=(IMAGE_HEIGHT,IMAGE_WIDTH)
                              )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False
    )
        


if __name__ == '__main__':
    dataloader()