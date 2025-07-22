import math, os
from pathlib import Path
from typing import List
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import numba as nb

# ------------------------------------------------------------------------------------
# Константы и подготовка
# ------------------------------------------------------------------------------------
SIZE = 64
dirs = np.array([
    (1,  0, 1.0), (-1,  0, 1.0), (0,  1, 1.0), (0, -1, 1.0),
    (1,  1, math.sqrt(2)), (1, -1, math.sqrt(2)),
    (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
], dtype=np.float32)

x_coords = np.repeat(np.arange(SIZE)[:, None], SIZE, 1) / (SIZE - 1)
y_coords = np.repeat(np.arange(SIZE)[None, :], SIZE, 0) / (SIZE - 1)
x_coords, y_coords = x_coords.flatten(), y_coords.flatten()   # (4096,)

# ------------------------------------------------------------------------------------
# 1. JIT-функция: строит рёбра для ОДНОЙ карты
# ------------------------------------------------------------------------------------

@nb.njit
def build_edges_single(map2d, dirs_arr):
    # 1) предвычислим количество узлов и макс. рёбер
    max_edges = map2d.size * dirs_arr.shape[0]
    src = np.empty(max_edges, np.int32)
    dst = np.empty(max_edges, np.int32)
    wts = np.empty(max_edges, np.float32)
    cnt = 0

    for x in range(SIZE):
        for y in range(SIZE):
            if map2d[x, y] != 0:
                continue

            base = x * SIZE + y
            for k in range(dirs_arr.shape[0]):
                dx = int(dirs_arr[k, 0])
                dy = int(dirs_arr[k, 1])
                nx = x + dx
                ny = y + dy

                # сначала проверяем границы
                if 0 <= nx < SIZE and 0 <= ny < SIZE and map2d[nx, ny] == 0:
                    src[cnt] = base
                    dst[cnt] = nx * SIZE + ny
                    wts[cnt] = dirs_arr[k, 2]
                    cnt += 1

    # обрежем до реального числа рёбер
    return src[:cnt], dst[:cnt], wts[:cnt]


# ------------------------------------------------------------------------------------
# 2. Внешняя функция: читает датасет, показывает tqdm, собирает Data-объекты
# ------------------------------------------------------------------------------------
def build_graphs_batch(maps, goals, starts, cf, q_tasks) -> List[Data]:
    graphs: List[Data] = []

    for j in tqdm(range(q_tasks), desc="Maps"):
        # --- рёбра (Numba) ---

        src, dst, w = build_edges_single(maps[j, 0], dirs)
        edge_index = torch.from_numpy(np.vstack((src, dst))).int()
        edge_attr  = torch.from_numpy(w).float()

        # --- признаки узлов ---
        walk   = maps[j, 0].ravel().astype(np.float32)
        goal   = goals[j, 0].ravel().astype(np.float32)
        start  = starts[j, 0].ravel().astype(np.float32)
        target = cf[j, 0].ravel().astype(np.float32)

        x_feat = np.stack([x_coords, y_coords, walk, goal, start], axis=1)
        x = torch.from_numpy(x_feat).float()  #.tensor(x_feat, dtype=torch.float32)
        y = torch.from_numpy(target).float()  #.tensor(target, dtype=torch.float32)

        graphs.append(Data(x=x,
                           edge_index=edge_index,
                           edge_attr=edge_attr,
                           y=y))

    return graphs

# ------------------------------------------------------------------------------------
# 3. Пример использования
# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Загрузка подготовленных массивов (пример для валидации)
    base = Path("/home/silvarum/TransPath_Adaptation/datasets/64/pfu/train")
    q_tasks = 96000
    maps   = np.load(base / "maps.npy")[0:q_tasks]
    goals  = np.load(base / "goals.npy")[0:q_tasks]
    starts = np.load(base / "starts.npy")[0:q_tasks]
    cf     = np.load(base / "cf.npy")[0:q_tasks]

    graphs = build_graphs_batch(maps, goals, starts, cf, q_tasks)

    save_path = "/home/silvarum/TransPath_Adaptation/gcn/datasets/pfu_small/train2.pt"
    torch.save(graphs, save_path)
    print(f"Saved {len(graphs)} graphs → {save_path}")
