import numpy as np
import os
import glob
from PIL import Image
from skimage.filters import threshold_otsu
import random


def generate_baldurs_gate(grid=128, input_dir="/home/silvarum/TransPath_Adaptation/bg512-png"):
    image_files = glob.glob(os.path.join(input_dir, "*.png"))
    random_file = random.choice(image_files)
    
    img = Image.open(random_file).convert("L").resize((grid, grid))
    image_array = np.asarray(img, dtype=np.float32)
    np.rot90(image_array, k= np.random.choice([1, 2, 3]))

    threshold = threshold_otsu(image_array)
    binary_map = np.zeros_like(image_array)
    binary_map[image_array > threshold] = 1.0
    
    return binary_map.astype(np.uint8)

