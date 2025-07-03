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
    src = []          # numba разберёт списки простых типов
    dst = []
    wts = []

    for x in range(SIZE):
        for y in range(SIZE):
            if map2d[x, y] != 1:            # непроходимая клетка
                continue
            idx = x * SIZE + y
            for k in range(dirs_arr.shape[0]):
                dx  = int(dirs_arr[k, 0])
                dy  = int(dirs_arr[k, 1])
                cost = dirs_arr[k, 2]

                nx, ny = x + dx, y + dy
                if map2d[nx, ny] == 1 and 0 <= nx < SIZE and 0 <= ny < SIZE:
                    src.append(idx)
                    dst.append(nx * SIZE + ny)
                    wts.append(cost)

    return (np.array(src, dtype=np.int32),
            np.array(dst, dtype=np.int32),
            np.array(wts, dtype=np.float32))

# ------------------------------------------------------------------------------------
# 2. Внешняя функция: читает датасет, показывает tqdm, собирает Data-объекты
# ------------------------------------------------------------------------------------
def build_graphs_batch(maps, goals, starts, cf) -> List[Data]:
    q_tasks = 64000
    graphs: List[Data] = []

    for j in tqdm(range(q_tasks), desc="Maps"):
        # --- рёбра (Numba) ---
        src, dst, w = build_edges_single(maps[j, 0], dirs)

        edge_index  = torch.tensor([src, dst], dtype=torch.long)
        edge_attr   = torch.tensor(w, dtype=torch.float32)

        # --- признаки узлов ---
        walk   = maps[j, 0].flatten().astype(np.float32)
        goal   = goals[j, 0].flatten().astype(np.float32)
        start  = starts[j, 0].flatten().astype(np.float32)
        target = cf[j, 0].flatten().astype(np.float32)

        x_feat = np.stack([x_coords, y_coords, walk, goal, start], axis=1)
        x = torch.tensor(x_feat, dtype=torch.float32)
        y = torch.tensor(target, dtype=torch.float32)

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
    base = Path("datasets/test/val")
    maps   = np.load(base / "maps.npy")
    goals  = np.load(base / "goals.npy")
    starts = np.load(base / "starts.npy")
    cf     = np.load(base / "cf.npy")

    graphs = build_graphs_batch(maps, goals, starts, cf)
    torch.save(graphs, "val_graphs.pt")
    print(f"Saved {len(graphs)} graphs → val_graphs.pt")
