import os
import logging
import time

import matplotlib.pyplot as plt
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import numpy as np


class Plotter:
    def __init__(self, output_dir, sub_dir='figures'):
        self.output_dir = output_dir
        self.sub_dir = sub_dir
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
            ax.set_ylim((0, max(ax.get_ylim()[1], 1)))
            ax.set_title(metric.capitalize())
            ax.legend()
        suffix = f' {name}' if name else ''
        fig.suptitle(f'History of Keras Model{suffix}')
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
        if store:
            self.store(fig, f'history-{suffix.replace(" ", "-")}', 'png', 'histories')
        return fig

    def plt_apply(self, axes, application, exceptions=[]):
        [application(ax) for ax in axes.flat if ax.get_title() not in exceptions]

    def plot_metrics(self, metrics, store=True):
        metrics = metrics.T  # MultiIndex Columns are more complicated for boxplot()
        metrics = metrics[metrics['rec'] != 0]  # Only executed runs
        runs = len(metrics.index.levels[1])  # Right now simply the number of predictors

        axes = metrics.boxplot(by='datasets', figsize=(13, 6), layout=(2, 3))
        # Bigger scale (but [0, 1] is not too hard to read)
        self.plt_apply(axes, lambda ax: ax.set_ylim((0, 0.6)))
        # Show random baseline
        self.plt_apply(axes, lambda ax: ax.axhline(0.33, color='gray', linestyle='--'),
                       ('mcc', 'f1'))
        plt.suptitle(f'Metrics grouped by datasets over {runs} runs')
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, hspace=0.4)
        fig = plt.gcf()
        if store:
            self.store(fig, f'boxplot_metrics_{runs}_runs', 'png')
        return fig

    def show_and_save_model(self, model):
        plot_model(model, to_file=os.path.join(self.output_dir, 'model.png'))
        return SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

    # --- Helper functions -------------------------------------------------- #

    def store(self, fig, title, extension='pdf', fig_dir=None, **kwargs):
        timestamp = time.strftime('%Y-%m-%d-%H%M%S')
        output_dir = os.path.join(self.output_dir, self.sub_dir)
        if fig_dir is not None:
            output_dir = os.path.join(output_dir, fig_dir)
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f'{title}-{timestamp}.{extension}')
        fig.savefig(path, **kwargs)
        self.logger.info(f'Stored plot at {path}')
