import numpy as np
from tqdm.auto import trange
import os


# Векторизованная функция для вычисления octile_distance
def octile_distance_vectorized(i_coords, j_coords, goal_i, goal_j):
    dx = np.abs(i_coords - goal_i)
    dy = np.abs(j_coords - goal_j)
    return np.maximum(dx, dy) + (np.sqrt(2) - 1) * np.minimum(dx, dy)


modes = ["test"]
main_dir = '/home/silvarum/TransPath_Adaptation/datasets/64/pfu'

data = np.load('/home/silvarum/TransPath_Adaptation/testset/testset.npy', mmap_mode='c')

maps = np.expand_dims(data[..., 0] == 0, 1)
starts = np.expand_dims(data[..., 3] == 1, 1)
goals = np.expand_dims(data[..., 1] == 1, 1)
abs_h = np.expand_dims(data[..., 2], 1)
focal = np.zeros_like(maps).astype(bool)

cf = np.zeros_like(maps).astype(float)
N = len(maps)
# Создаем матрицы координат один раз (не меняются между итерациями)
i_coords, j_coords = np.indices((64, 64))


for idx in trange(N):
    goal_i, goal_j = np.where(goals[idx, 0])
    goal_i, goal_j = goal_i[0], goal_j[0]
    cf[idx, 0, goal_i, goal_j] = 1.0
    
    # Вычисляем расстояния для всех точек сразу
    distances = octile_distance_vectorized(i_coords, j_coords, goal_i, goal_j)
    
    # Создаем маску для abs_h > 0
    mask = abs_h[idx, 0] > 0
    
    # Вычисляем отношение только там, где маска истинна
    cf[idx, 0, mask] = distances[mask] / abs_h[idx, 0, mask]


for mode in modes:
    os.makedirs(f'{main_dir}/{mode}', exist_ok=True)

    np.save(f'{main_dir}/{mode}/maps.npy', maps)
    np.save(f'{main_dir}/{mode}/starts.npy', starts)
    np.save(f'{main_dir}/{mode}/goals.npy', goals)
    np.save(f'{main_dir}/{mode}/focal.npy', focal)
    np.save(f'{main_dir}/{mode}/cf.npy', cf)
    np.save(f'{main_dir}/{mode}/abs.npy', abs_h)