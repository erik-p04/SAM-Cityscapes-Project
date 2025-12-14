import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

CITYSCAPES_IDS = {
    "road": 7,
    "sidewalk": 8,
    "building": 11,
    "person": 24,
    "car": 26,
}

class CityscapesSemanticDataset(Dataset):
    def __init__(self, root, image_dir, mask_dir, classes):
        self.image_dir = os.path.join(root, image_dir)
        self.mask_dir = os.path.join(root, mask_dir)
        self.classes = classes
        self.class_map = {cid: i for i, cid in enumerate(CITYSCAPES_IDS.values())}

        self.samples = []
        for city in os.listdir(self.image_dir):
            img_folder = os.path.join(self.image_dir, city)
            for f in os.listdir(img_folder):
                if f.endswith("_leftImg8bit.png"):
                    mask = f.replace("leftImg8bit", "gtFine_labelIds")
                    self.samples.append((
                        os.path.join(img_folder, f),
                        os.path.join(self.mask_dir, city, mask)
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        target = np.full(mask.shape, -1, dtype=np.int64)

        for cid, new_id in self.class_map.items():
            target[mask == cid] = new_id

        target = torch.from_numpy(target)

        return image, target
