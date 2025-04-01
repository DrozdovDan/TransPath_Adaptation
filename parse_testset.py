import numpy as np
from tqdm.auto import trange
from analysis.astar import octile_distance
import os

main_dir = './testset'

data = np.load(f'{main_dir}/testset.npy', mmap_mode='c')

maps = np.expand_dims(data[..., 0] == 0, 1)
starts = np.expand_dims(data[..., 3] == 1, 1)
goals = np.expand_dims(data[..., 1] == 1, 1)
abs_h = np.expand_dims(data[..., 2], 1)
focal = np.zeros_like(maps).astype(bool)

cf = np.zeros_like(maps).astype(float)

N = len(maps)

for idx in trange(N):
    goal_i, goal_j = np.where(goals[idx, 0])
    goal_i, goal_j = goal_i[0], goal_j[0]
    cf[idx, 0, goal_i, goal_j] = 1.0
    for i in range(64):
        for j in range(64):
            if abs_h[idx, 0, i, j] > 0:
                cf[idx, 0, i, j] = octile_distance(i, j, goal_i, goal_j) / abs_h[idx, 0, i, j]

os.makedirs(f'{main_dir}/test', exist_ok=True)

np.save(f'{main_dir}/test/maps.npy', maps)
np.save(f'{main_dir}/test/starts.npy', starts)
np.save(f'{main_dir}/test/goals.npy', goals)
np.save(f'{main_dir}/test/focal.npy', focal)
np.save(f'{main_dir}/test/cf.npy', cf)
np.save(f'{main_dir}/test/abs.npy', abs_h)