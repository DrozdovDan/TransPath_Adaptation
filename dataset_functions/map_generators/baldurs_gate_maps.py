import numpy as np
import random

def generate_baldurs_gate():
    return np.load("/home/silvarum/TransPath_Adaptation/test_with_rotations.npy")[random.randint(0, 299)]
