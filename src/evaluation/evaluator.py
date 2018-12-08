# import gc
import logging
import os
import pickle
# import traceback

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from sklearn.metrics import classification_report, matthews_corrcoef,\
    accuracy_score, precision_recall_fscore_support

from .config import init_logging
from ..preparation import prepare_data
from ..pipeline import run_pipeline
from ..algorithms import Algorithm
from ..utils import create_deep_dict


class Evaluator:
    def __init__(self, name: str, datasets: list, get_predictors: callable, output_dir: {str}=None,
                 seed: int=None, create_log_file: bool=True, store: bool=False,
                 n_train_samples: int=60000, n_test_samples: int=6000):
        """
        :param datasets: list of datasets
        :param predictors: callable that returns list of predictors
        """
        self.name = name
        self.datasets = datasets
        self.dataset_names = [str(x) for x in self.datasets]
        assert np.unique(self.dataset_names).size == len(self.dataset_names),\
            'Some datasets have the same name!'
        self.get_predictors = get_predictors
        self.predictor_names = [str(x) if isinstance(x, Algorithm) else x.__class__.__name__
                                for x in self.get_predictors(1)]
        assert np.unique(self.predictor_names).size == len(self.predictor_names),\
            'Some predictors have the same name!'
        self.output_dir = output_dir or 'reports'
        self.results = None
        if create_log_file:
            init_logging(os.path.join(self.output_dir, 'logs'))
        self.logger = logging.getLogger(__name__)
        self.metrics = None
        self.seed = seed
        self.store = store
        self.n_train_samples = n_train_samples
        self.n_test_samples = n_test_samples
        self._temp_pipelines = create_deep_dict(self.dataset_names, self.predictor_names)

    def export_results(self):
        output_dir = os.path.join(self.output_dir, 'evaluators')
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime('%Y-%m-%d-%H%M%S')
        path = os.path.join(output_dir, f'{self.name}-{timestamp}.pkl')
        self.logger.info(f'Store evaluator results at {os.path.abspath(path)}')
        save_dict = {
            # TODO: 'pipelines': map_deep_dict(self._temp_pipelines, 2, lambda x: x[0]),
            # If not working then: x[0].steps[-1][1].history
            'dataset_names': self.dataset_names,
            'predictor_names': self.predictor_names,
            'metrics': self.metrics,
            'results': self.results,
            'output_dir': self.output_dir,
            'seed': int(self.seed),
        }
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        return path

    def measure_pipeline_run_results(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        acc = accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        target_names = np.array(['Down', 'Still', 'Up'])[np.unique(y_true).astype(int) + 1]
        self.logger.debug(
            f'\n{classification_report(y_true, y_pred, target_names=target_names)}\n\nMCC='
            f'{mcc:.5f}, Accuracy={acc:.5f}, Precision={prec:.5f}, Recall={rec:.5f}, F1={f1:.5f}')
        return {
            'prec': round(prec, 5),
            'rec': round(rec, 5),
            'f1': round(f1, 5),
            'acc': round(acc, 5),
            'mcc': round(mcc, 5)
        }

    def __call__(self):
        multiindex = pd.MultiIndex.from_product([self.dataset_names, self.predictor_names],
                                                names=['datasets', 'predictors'])
        # metrics = create_deep_dict(self.dataset_names, self.predictor_names)
        metrics = pd.DataFrame(0, index=['prec', 'rec', 'f1', 'acc', 'mcc'], columns=multiindex)
        metrics.sort_index(axis=1, inplace=True)
        # Might be required if datasets have different sizes
        # predictions = create_deep_dict(self.dataset_names, self.predictor_names)
        predictions = pd.DataFrame(0, index=range(self.n_test_samples), columns=multiindex)
        for ds in self.datasets:
            self.logger.info(f"{'-'*10} Prepare dataset {'-'*10}")
            data = prepare_data(ds, self.n_train_samples, self.n_test_samples)
            n_features = len(data[0].columns.levels[1])
            predictors = self.get_predictors(n_features)
            for predictor, predictor_name in zip(predictors, self.predictor_names):
                can_handle_time_dim = isinstance(predictor, Algorithm) and\
                    predictor.can_handle_time_dim()
                self.logger.info(f"{'-'*10} {predictor_name} | {ds} {'-'*10}")
                pipeline, y_pred = run_pipeline(predictor, data, time_dim=can_handle_time_dim)
                time.sleep(1)
                y_pred = y_pred.clip(-1, 1)
                ev = self.measure_pipeline_run_results(data[3], y_pred)
                predictions.loc[:, (str(ds), predictor_name)] = y_pred
                assert all(metrics.index == list(ev.keys()))
                metrics.loc[:, (str(ds), predictor_name)] = ev.values()
                self._temp_pipelines[str(ds)][predictor_name] = pipeline, y_pred
        self.predictions = predictions
        self.metrics = metrics
        if self.store:
            self.export_results()
        return self.metrics

    def get_metrics(self):
        return self.metrics

    def get_mcc_metric(self):
        metrics = self.get_metrics()
        mcc_flatten = metrics.loc['mcc'].reset_index()
        mcc = mcc_flatten.pivot(*mcc_flatten.columns)  # index, column, value
        return mcc

    # @property
    # def predictors(self):
    #     predictors = self._predictors(self.seed)
    #     assert np.unique([str(x) for x in predictors]).size == len(predictors),\
    #         'Some predictors have the same name!'
    #     return predictors

    # Import benchmark_results if this evaluator uses the same detectors and datasets
    # self.results are not available because they are overwritten by each run
    # def import_results(self, name):
    #     output_dir = os.path.join(self.output_dir, 'evaluators')
    #     path = os.path.join(output_dir, f'{name}.pkl')
    #     self.logger.info(f'Read evaluator results at {os.path.abspath(path)}')
    #     with open(path, 'rb') as f:
    #         save_dict = pickle.load(f)
    #
    #     self.logger.debug(f'Importing detectors {"; ".join(save_dict["detectors"])}')
    #     my_detectors = [str(x) for x in self.detectors]
    #     assert np.array_equal(save_dict['detectors'], my_detectors),\
    #         'Detectors should be the same'
    #
    #     self.logger.debug(f'Importing datasets {"; ".join(save_dict["datasets"])}')
    #     my_datasets = [str(x) for x in self.datasets]
    #     assert np.array_equal(save_dict['datasets'], my_datasets), 'Datasets should be the same'
    #
    #     self.benchmark_results = save_dict['benchmark_results']
    #     self.seed = save_dict['seed']
    #     self.results = save_dict['results']
