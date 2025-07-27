import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astar import global_octile_distance, astar, Map, SearchTreePQD, make_path, draw_simple, draw, Node, multi_global_octile_distance
import math
from metrics_collector import *
from model import *
import scipy.stats as ss

checkpoint = torch.load('/home/silvarum/TransPath_Adaptation/checkpoints/model_name=TransPathModel, ds=/home/silvarum/TransPath_Adaptation/datasets/512/beta, bs=25, ep=100, lr=0.00032, OneCycle=False, skip=True, downsample_steps=6, embeddings=False-ep2-0.01076-date:27072025_152308.ckpt', weights_only=False)

from collections import OrderedDict
weights = OrderedDict({k[6:] : w for k, w in checkpoint['state_dict'].items()})

model = TransPathModel(resolution=(512, 512), skip=True, downsample_steps=6)

model.load_state_dict(weights)

torch.save(model.state_dict(), '../weights/beta_512_2')

cells, starts, goals, cfs = data_from_dir('/home/silvarum/TransPath_Adaptation/datasets/512/pfu/test')

kwargs = {
    'skip' : True,
    'downsample_steps' : 6,
    'resolution' : (512, 512)
}

model = create_TransPath_model(model_name=TransPathModel, weights_path='../weights/beta_512_2', device='cuda:1', **kwargs)

baseline = pd.read_csv('pfu_512.csv')

res = contain_ratios(cells, starts, goals, model=model, node_type='optimal', baseline=baseline, threshold=1.05, verbose=2)

import pickle

print(len(res['model']))


# Save to a pickle file
with open("../results/beta_512_2.pkl", "wb") as f:
    pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)

import pickle

# Save to a pickle file
with open("../results/beta_512_2.pkl", "rb") as f:
    res = pickle.load(f)

res = get_metrics(res)

print(res)