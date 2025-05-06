import os
import numpy as np
from scipy.io import loadmat
from htc import DataPath

class DataLoader:
    def __init__(self, base_path):
        self.base_path = base_path

    def load_cube(self, image_name):
        path = DataPath.from_image_name(image_name)
        cube = path.read_cube()
        return cube

    def load_signatures(self, hb_sign_path, hbo2_sign_path):
        hb_sign = np.loadtxt(hb_sign_path)[:, 1]
        hbo2_sign = np.loadtxt(hbo2_sign_path)[:, 1]
        return hb_sign, hbo2_sign

    def load_mask(self, mask_path):
        if os.path.exists(mask_path):
            mask = np.load(mask_path)
            return mask
        else:
            return None
