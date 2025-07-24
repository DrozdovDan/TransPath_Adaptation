import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astar import global_octile_distance, astar, Map, SearchTreePQD, make_path, draw_simple, draw, Node, multi_global_octile_distance
import math
from metrics_collector import *
from model import *
import scipy.stats as ss

checkpoint = torch.load('/home/silvarum/TransPath_Adaptation/checkpoints/ds=/home/silvarum/TransPath_Adaptation/datasets/128/betanoised_figures, bs=256, ep=100, lr=0.008, OneCycle=True, skip=True, downsample_steps=4-ep93-0.00028-date:21072025_005553.ckpt', weights_only=False)

from collections import OrderedDict
weights = OrderedDict({k[6:] : w for k, w in checkpoint['state_dict'].items()})

model = TransPathModel(resolution=(128, 128), skip=True, downsample_steps=4)

model.load_state_dict(weights)

torch.save(model.state_dict(), '../weights/beta_fig_128')