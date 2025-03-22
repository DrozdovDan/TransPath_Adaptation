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
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from model import TransPathModel, GridData

from astar import wastar, cfastar, astar_func, Map, global_octile_distance, SearchTreePQD, make_path, draw_simple, Node
import os

def cfs_from_file(file_path: str):
    return np.load(file_path, mmap_mode='c')

def cells_from_file(file_path: str):
    return np.load(file_path, mmap_mode='c')

def starts_from_file(file_path: str):
    data = np.load(file_path, mmap_mode='c')
    return np.argwhere(data)[:, 2:]

def goals_from_file(file_path: str):
    data = np.load(file_path, mmap_mode='c')
    return np.argwhere(data)[:, 2:]

def data_from_dir(dir_path: str, maps_filename: str='maps.npy', starts_filename: str='starts.npy', goals_filename: str='goals.npy', cfs_filename: str=None):
    maps_path = os.path.join(dir_path, maps_filename)
    starts_path = os.path.join(dir_path, starts_filename)
    goals_path = os.path.join(dir_path, goals_filename)
    if cfs_filename:
        cfs_path = os.path.join(dir_path, cfs_filename)
        return cells_from_file(maps_path), starts_from_file(starts_path), goals_from_file(goals_path), cfs_from_file(cfs_path)
    return cells_from_file(maps_path), starts_from_file(starts_path), goals_from_file(goals_path), None

def astar_octile_search(cells: np.ndarray, starts: np.ndarray, goals: np.ndarray, verbose: int=1):
    assert cells.shape[0] == starts.shape[0] == goals.shape[0]

    metrics = {'index' : [], 'path_length' : [], 'expanded_nodes_num' : []}
    nonexistent_paths = []
    iterator = range(cells.shape[0])
    if verbose > 0:
        iterator = trange(cells.shape[0], desc='A* octile search')
    for i in iterator:
        grid = Map(cells[i, 0])
        result = astar_func(grid, *starts[i], *goals[i], global_octile_distance, SearchTreePQD)
        if not result[0]:
            nonexistent_paths.append(i)
            if verbose > 1:
                draw_simple(grid, *starts[i], *goals[i], None, result[-2], result[-1])
            continue
        path, length = make_path(result[1])
        metrics['index'].append(i)
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

def wastar_octile_search(cells: np.ndarray, starts: np.ndarray, goals: np.ndarray, w: float=2.0, verbose: int=1):
    assert cells.shape[0] == starts.shape[0] == goals.shape[0]

    metrics = {'index' : [], 'path_length' : [], 'expanded_nodes_num' : []}
    nonexistent_paths = []
    iterator = range(cells.shape[0])
    if verbose > 0:
        iterator = trange(cells.shape[0], desc='WA* octile search')
    for i in iterator:
        grid = Map(cells[i, 0])
        heuristic = global_octile_distance(*grid.get_size(), *goals[i])
        result = wastar(grid, *starts[i], *goals[i], heuristic, w, SearchTreePQD)
        if not result[0]:
            nonexistent_paths.append(i)
            if verbose > 1:
                draw_simple(grid, *starts[i], *goals[i], None, result[-2], result[-1])
            continue
        path, length = make_path(result[1])
        metrics['index'].append(i)
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

def cfastar_octile_search(cells: np.ndarray, starts: np.ndarray, goals: np.ndarray, cfs: np.ndarray, verbose: int=1):
    assert cells.shape[0] == starts.shape[0] == goals.shape[0]

    metrics = {'index' : [], 'path_length' : [], 'expanded_nodes_num' : []}
    nonexistent_paths = []
    iterator = range(cells.shape[0])
    if verbose > 0:
        iterator = trange(cells.shape[0], desc='A* with cf octile search')
    for i in iterator:
        grid = Map(cells[i, 0])
        heuristic = global_octile_distance(*grid.get_size(), *goals[i])
        result = cfastar(grid, *starts[i], *goals[i], heuristic, cfs[i, 0], SearchTreePQD)
        if not result[0]:
            nonexistent_paths.append(i)
            if verbose > 1:
                draw_simple(grid, *starts[i], *goals[i], None, result[-2], result[-1])
            continue
        path, length = make_path(result[1])
        metrics['index'].append(i)
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

def create_TransPath_model(model_name=TransPathModel, weights_path: str=None, device: str='cpu'):
    torch_device = torch.device(device)
    model = model_name()
    if weights_path:
        model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.to(torch_device)
    return model

class GridDataTest(Dataset):
    def __init__(self, cells: np.ndarray, starts: np.ndarray, goals: np.ndarray, img_h=64, img_w=64):
        self.img_h = img_h
        self.img_w = img_w

        self.maps   = cells
        starts_grid = np.zeros_like(cells)
        starts_grid[np.arange(cells.shape[0]), 0, starts[:, 0], starts[:, 1]] = 1
        goals_grid = np.zeros_like(cells)
        goals_grid[np.arange(cells.shape[0]), 0, goals[:, 0], goals[:, 1]] = 1
        self.goals  = goals_grid
        self.starts = starts_grid


    def __len__(self):
        return len(self.maps)
    
    def __getitem__(self, idx):
        return (torch.from_numpy(self.maps[idx].astype('float32')), 
                torch.from_numpy(self.starts[idx].astype('float32')), 
                torch.from_numpy(self.goals[idx].astype('float32')))

def cfastar_octile_search_with_prediction(cells: np.ndarray, starts: np.ndarray, goals: np.ndarray, model: nn.Module, save_predictions_to: str=None, verbose: int=1):
    dataset = GridDataTest(cells, starts, goals, img_h=cells.shape[-2], img_w=cells.shape[-1])
    dataloader = DataLoader(
        dataset, 
        batch_size=1,
        shuffle=False, 
        pin_memory=True,
        drop_last=False
    )
    iterator = dataloader
    if verbose > 0:
        iterator = tqdm(dataloader, desc='Predicting heuristics')
    predictions = []
    model.eval()
    for map_design, start, goal in iterator:
        inputs = torch.cat([map_design, goal], dim=1)
        inputs = inputs.to(next(model.parameters()).device)
        
        with torch.no_grad():
            prediction = (model(inputs) + 1) / 2

        predictions.append(prediction)

    predictions = torch.cat(predictions).cpu().detach().numpy()

    if save_predictions_to:
        np.save(save_predictions_to, predictions)
    
    return cfastar_octile_search(cells, starts, goals, predictions, verbose=verbose)

def contain_ratios(cells: np.ndarray, starts: np.ndarray, goals: np.ndarray, ws: list[int]=None, cfs: np.ndarray=None, model: nn.Module=None, save_predictions_to: str=None, verbose: int=1):
    assert (cfs is not None) ^ (model is not None)

    baseline = astar_octile_search(cells, starts, goals, verbose=verbose)

    model_df = None
    
    if model:
        model_df = cfastar_octile_search_with_prediction(cells, starts, goals, model, save_predictions_to, verbose=verbose)
    else:
        model_df = cfastar_octile_search(cells, starts, goals, cfs, verbose=verbose)

    w_dfs = {}

    if ws:
        iterator = ws
        if verbose > 0:
            iterator = tqdm(ws, desc='Computing WA* statistics')
        for w in iterator:
            w_dfs[w] = wastar_octile_search(cells, starts, goals, w, verbose=verbose)

    baseline_array = baseline[['path_length', 'expanded_nodes_num']].to_numpy()
    model_array = model_df[['path_length', 'expanded_nodes_num']].to_numpy()

    w_arrays = {}

    if w_dfs:
        w_arrays = {w : df[['path_length', 'expanded_nodes_num']].to_numpy() for w, df in w_dfs.items()}

    results = {}

    results['baseline'] = pd.DataFrame(baseline_array / baseline_array, columns=['path_length', 'expanded_nodes_num'])
    results['model'] = pd.DataFrame(model_array / baseline_array, columns=['path_length', 'expanded_nodes_num'])

    if w_arrays:
        results.update({f'w={w}' : pd.DataFrame(array / baseline_array, columns=['path_length', 'expanded_nodes_num']) for w, array in w_arrays.items()})

    return results

    


    
        