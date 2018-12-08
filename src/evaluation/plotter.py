import os
import logging
import time

import matplotlib.pyplot as plt
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import numpy as np


class Plotter:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)

    # --- Final plot functions ---------------------------------------------- #

    def plot_history(self, history, name=None, store=True):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        epochs = len(history.history['loss'])

        metrics = []
        for i, label in enumerate(sorted(history.history)):
            if label[:4] != 'val_':
                metrics.append(label)
                ax = axes[len(metrics)-1]
                linestyle = None
            else:
                ax = axes[i-len(metrics)]
                linestyle = 'dashed'
            ax.plot(np.arange(1, epochs+1), history.history[label], label=label,
                    linestyle=linestyle)

        for metric, ax in zip(metrics, axes):
            ax.set_xlabel('epoch')
            ax.set_xlim((1, epochs))
            ax.set_title(metric.capitalize())
            ax.legend()
        fig.tight_layout()
        suffix = f' {name}' if name else ''
        fig.suptitle(f'History of Keras Model{suffix}')
        fig.subplots_adjust(top=0.85, hspace=1, right=1, left=0)
        if store:
            suffix = f'-{name}' if name else ''
            self.store(fig, f'history{suffix}', 'png')

    def show_and_save_model(self, model):
        plot_model(model, to_file=os.path.join(self.output_dir, 'model.png'))
        return SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

    # --- Helper functions -------------------------------------------------- #

    def store(self, fig, title, extension='pdf', **kwargs):
        timestamp = time.strftime('%Y-%m-%d-%H%M%S')
        output_dir = os.path.join(self.output_dir, 'figures')
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f'{title}-{timestamp}.{extension}')
        fig.savefig(path, **kwargs)
        self.logger.info(f'Stored plot at {path}')
