import os, json
from loguru import logger
import cv2
import numpy as np
from torch.utils.data import Dataset

class TargetDataset(Dataset):
    def __init__(self, dataset_path, transform = None, test_transform = None, is_test = False):
        super().__init__()
        logger.info("Dataset path:", dataset_path)
        
        # annotations.json -> ["filename", class_id(int64)], ...
        annotation_path = os.path.join(dataset_path, "annotations.json")
        self.annotations = json.load(open(annotation_path))
        self.jpeg_basepath = os.path.join(dataset_path, "Main")
        
        self.transform = transform
        self.test_transform = test_transform
        self.is_test = is_test
        
    def __getitem__(self, idx):
        image_name, annotation = self.annotations[idx]
        image_path = os.path.join(self.jpeg_basepath, image_name)
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # H, W, C
        
        if self.is_test:
            if self.test_transform:
                image = self.test_transform(image=image)['image']
        elif self.transform is not None:
            image = self.transform(image=image)['image']
        
        image = np.transpose(image, (2, 0, 1)) # C, H, W
        image = np.divide(image, 255.0, dtype=np.float32)
        
        annotation = np.array(annotation, dtype=np.int64)
        return image, annotation
    
    def __len__(self):
        return len(self.annotations)