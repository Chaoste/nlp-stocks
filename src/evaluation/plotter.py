import os
import pickle
import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Plotter:
    def __init__(self, output_dir, pickle_dirs=None, dataset_names=None, detector_names=None):
        self.output_dir = output_dir
        self.dataset_names = dataset_names
        self.detector_names = detector_names
        self.results = None
        self.logger = logging.getLogger(__name__)
        if pickle_dirs is not None:
            self.results = self.import_results_for_runs(pickle_dirs)

    # --- Final plot functions ----------------------------------------------- #

    # --- Helper functions --------------------------------------------------- #

    def store(self, fig, title, extension='pdf', **kwargs):
        timestamp = time.strftime('%Y-%m-%d-%H%M%S')
        output_dir = os.path.join(self.output_dir, 'figures')
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f'{title}-{timestamp}.{extension}')
        fig.savefig(path, **kwargs)
        self.logger.info(f'Stored plot at {path}')
