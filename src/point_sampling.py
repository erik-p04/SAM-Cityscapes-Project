import numpy as np
from scipy.ndimage import distance_transform_edt

# Sampling center-of-mass prompts

def get_center_point(class_mask):
    if class_mask.sum() == 0:
        return None

    dist = distance_transform_edt(class_mask)
    y, x = np.unravel_index(dist.argmax(), dist.shape)
    return np.array([[x, y]]), np.array([1])
