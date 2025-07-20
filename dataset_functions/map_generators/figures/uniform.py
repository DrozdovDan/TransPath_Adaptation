import random
import numpy as np

def generate_uniform(grid):
    # Первый проход
    uniform = np.random.choice([0, 1], size=(grid, grid))

    # Второй
    for _ in range(20):
        trap_x = random.randint(0, 63)
        trap_y = random.randint(0, 63)
        uniform[trap_x, trap_y] = 0
    return uniform