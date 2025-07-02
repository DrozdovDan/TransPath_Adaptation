import random
import numpy as np

def generate_dumbQRs():
    # Первый проход
    dumbQR = np.random.choice([0, 1], size=(64, 64))

    # Второй
    for _ in range(20):
        trap_x = random.randint(0, 63)
        trap_y = random.randint(0, 63)
        dumbQR[trap_x, trap_y] = 0
    return dumbQR