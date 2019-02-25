import os
import re
import tracemalloc

import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook as tqdm

import sys
sys.path.append("..") # Adds higher directory to python modules path for importing from src dir

tracemalloc.start(10)

__all__ = [
    'os', 're', 'tracemalloc', 'sys',
    'pd', 'np', 'matplotlib', 'plt', 'tqdm', 
    'init', 'reset_all_notebook_vars', 'inspect_snapshot_diff', 'display_top'
]


def init():
    print("Execute the following jupyter commands manually:")
    print("%matplotlib inline")
    print("%load_ext autotime")
    print("%load_ext autoreload")
    print("%autoreload 2")

    
def reset_all_notebook_vars(*varnames):
    # varnames are what you want to keep
    globals_ = globals()
    to_save = {v: globals_[v] for v in varnames}
    to_save['reset_all_notebook_vars'] = reset_all_notebook_vars  # lets keep this function by default
    del globals_
    get_ipython().magic("reset")
    globals().update(to_save)

    
def inspect_snapshot_diff(first_snapshot):
    second_snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    display_top(top_stats, limit=5)
    

def display_top(top_stats, limit=3):
    # snapshot = snapshot.filter_traces((
    #     tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
    #     tracemalloc.Filter(False, "<unknown>"),
    # ))
    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))