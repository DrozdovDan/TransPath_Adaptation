import random
import traceback
from heapq import heappop, heappush
from pathlib import Path
from textwrap import dedent
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw
import pandas as pd
from tqdm.auto import tqdm ,trange

from astar import wastar, cfastar, astar_func, Map, global_octile_distance, SearchTreePQD, make_path, draw_simple, Node
import os

def cf_from_file(file_path: str):
    return np.load(file_path, mmap_mode='c')

def cells_from_file(file_path: str):
    return np.load(file_path, mmap_mode='c')

def starts_from_file(file_path: str):
    data = np.load(file_path, mmap_mode='c')
    return np.argwhere(data)[:, 2:]

def goals_from_file(file_path: str):
    data = np.load(file_path, mmap_mode='c')
    return np.argwhere(data)[:, 2:]

def data_from_dir(dir_path: str, maps_filename: str='maps.npy', starts_filename: str='starts.npy', goals_filename: str='goals.npy', cfs_filename: str='cf.npy'):
    maps_path = os.path.join(dir_path, maps_filename)
    starts_path = os.path.join(dir_path, starts_filename)
    goals_path = os.path.join(dir_path, goals_filename)
    cfs_path = os.path.join(dir_path, cfs_filename)
    return cells_from_file(maps_path), starts_from_file(starts_path), goals_from_file(goals_path), cf_from_file(cfs_path)

def astar_octile_search(cells: np.ndarray, starts: np.ndarray, goals: np.ndarray, verbose: int=1):
    assert cells.shape[0] == starts.shape[0] == goals.shape[0]

    metrics = {'path_length' : [], 'expanded_nodes_num' : []}
    nonexistent_paths = []
    iter_func = range
    if verbose > 0:
        iter_func = trange
    for i in iter_func(cells.shape[0]):
        grid = Map(cells[i, 0])
        result = astar_func(grid, *starts[i], *goals[i], global_octile_distance, SearchTreePQD)
        if not result[0]:
            metrics['path_length'].append(None)
            metrics['expanded_nodes_num'].append(None)
            nonexistent_paths.append(i)
            if verbose > 1:
                draw_simple(grid, *starts[i], *goals[i], None, result[-2], result[-1])
            continue
        path, length = make_path(result[1])
        metrics['path_length'].append(length)
        metrics['expanded_nodes_num'].append(len(result[-1]))
        if verbose > 1:
            draw_simple(grid, *starts[i], *goals[i], path, result[-2], result[-1])
    if verbose > 0:
        print(f'During the search was discovered {len(nonexistent_paths)} non-existent paths')
        if nonexistent_paths:
            print(f'Indices of tasks with non-existent paths:')
            print(*nonexistent_paths)
    
    return pd.DataFrame.from_dict(metrics)