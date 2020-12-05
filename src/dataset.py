import torch 
import numpy as np
from albumentations import Compose, Normalize, MedianBlur, GaussianBlur, \
    MotionBlur, GaussNoise, MultiplicativeNoise, Cutout, \
    CoarseDropout, GridDistortion, ElasticTransform, \
    RandomBrightness, RandomContrast, RandomBrightnessContrast, \
    ShiftScaleRotate, IAAPiecewiseAffine

from PIL import Image

class OcrDataset: 
    """
    Handle images for dataloader.
    Apply resize and augmentations.
    Put attention to resize PIL: first goes width then height
    """
    
    def __init__(self, image_path, labels, resize=None):
        self.image_path = image_path
        self.labels = labels
        self.resize = resize
        self.augmentations = Compose(
            [
            Normalize(always_apply=True),
            # MedianBlur(blur_limit=5, p=0.2),
            # GaussianBlur(p=0.2),
            # MotionBlur(p=0.2),
            # GaussNoise(var_limit=5. / 255., p=0.2),
            # MultiplicativeNoise(p=0.2),
            # Cutout(num_holes=8, max_h_size=10, max_w_size=10,p=0.2),
            # CoarseDropout(max_holes=8, max_height=10, max_width=10,p=0.2),
            # GridDistortion(p=0.2),
            # ElasticTransform(sigma=50, alpha=1, alpha_affine=10, p=0.2),
            # RandomBrightness(p=0.2),
            # RandomContrast(p=0.2),
            # RandomBrightnessContrast(p=0.2),
            # IAAPiecewiseAffine(p=0.2),
            # ShiftScaleRotate(shift_limit=0.1,
            #                     scale_limit=0.1,
            #                     rotate_limit=30,
            #                     p=0.2)             
            ]                                    
                                            
                                            )
        
    def __len__(self):
        return len(self.image_path)
        
    def __getitem__(self, item):
        image = Image.open(self.image_path[item]).convert('RGB')
        labels = self.labels[item]
        
        if self.resize is not None:
            image = image.resize((self.resize[1], self.resize[0]), resample=Image.BILINEAR)
            
        
        image = np.array(image)
        augmented_image = self.augmentations(image=image)
        image = augmented_image["image"]
        image = np.transpose(image,(2,0,1)).astype(np.float32)
        
        return {
            "images": torch.tensor(image, dtype=torch.float),
            "labels": torch.tensor(labels, dtype=torch.long)
        }