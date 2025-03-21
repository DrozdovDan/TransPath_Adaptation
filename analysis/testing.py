import numpy as np
import pandas as pd
from astar import global_octile_distance
from metrics_collector import starts_from_file, data_from_dir, astar_octile_search, cfastar_octile_search

cells, starts, goals, cfs = data_from_dir('../islands/test')

print(cfastar_octile_search(cells, starts, goals, cfs, verbose=2))