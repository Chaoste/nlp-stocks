import copy

import pandas as pd


def announce_experiment(title: str, dashes: int = 70):
    print(f'\n###{"-"*dashes}###')
    message = f'Experiment: {title}'
    before = (dashes - len(message)) // 2
    after = dashes - len(message) - before
    print(f'###{"-"*before}{message}{"-"*after}###')
    print(f'###{"-"*dashes}###\n')


def create_deep_dict(*layers, fill=None):
        for layer in reversed(layers):
            fill = dict([(k, copy.deepcopy(fill) if not callable(fill)
                         else fill(k)) for k in layer])
        return fill


def map_deep_values(v, n_layers, mapping):
    assert n_layers >= 0
    if n_layers == 0:
        return mapping(v)
    else:
        return dict([(k, map_deep_values(v[k], n_layers - 1, mapping)) for k in v])


# Was required for transforming deep dict into MultiIndex-DataFrame
def _merge_metrics(metrics_per_run):
    reform = {(level1_key, level2_key): level3_dict
              for level1_key, level2_dict in metrics_per_run.items()
              for level2_key, level3_dict in level2_dict.items()}
    all_metrics = pd.DataFrame(reform)
    all_metrics.columns.set_names(['dataset', 'predictor'], inplace=True)
    mcc_metric = pd.DataFrame(all_metrics.loc['mcc']).reset_index()
    mcc_metric = mcc_metric.pivot(*mcc_metric.columns)  # index, columns, values
    return all_metrics, mcc_metric
