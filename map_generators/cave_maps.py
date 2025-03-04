import numpy as np
from map_generators.island_maps import generate_islands


# Функция сглаживания карты
def random_walker(map_of_islands):
    deltas = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    for _ in range(8):
        walls_sum = np.zeros_like(map_of_islands, dtype=np.float32)
        for dx, dy in deltas:
            shifted = np.roll(map_of_islands, shift=dy, axis=0)
            shifted = np.roll(shifted, shift=dx, axis=1)
            walls_sum += (shifted == 0)
        window_sum = len(deltas)
        condition = (walls_sum / window_sum) < 0.5
        map_of_islands[condition] = 1
    return map_of_islands


# Пещеры –- это сглаженные острова!
def generate_caves():
    return random_walker(generate_islands())
