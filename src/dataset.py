import os
import cv2
import numpy as np
from torch.utils.data import Dataset

# Cityscapes loader

CITYSCAPES_IDS = {
    "road": 7,
    "sidewalk": 8,
    "building": 11,
    "person": 24,
    "car": 26,
}

class CityscapesDataset(Dataset):
    def __init__(self, root, image_dir, mask_dir, classes):
        self.root = root
        self.image_dir = os.path.join(root, image_dir)
        self.mask_dir = os.path.join(root, mask_dir)
        self.classes = classes

        self.images = []
        for city in os.listdir(self.image_dir):
            img_folder = os.path.join(self.image_dir, city)
            for f in os.listdir(img_folder):
                if f.endswith("_leftImg8bit.png"):
                    mask_path = os.path.join(
                        self.mask_dir, city, f.replace("leftImg8bit", "gtFine_labelIds")
                    )
                    self.images.append((os.path.join(img_folder, f), mask_path))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img_path, mask_path = self.images[i]
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)  # labelIds mask
        return image, mask, img_path
