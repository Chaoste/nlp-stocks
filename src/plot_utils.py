import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import scipy.stats as scs
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt

import src.math_utils as math_utils


def compare_with_normal(data, title=None, **kwargs):
    pd.DataFrame(data).hist(bins=100, density=True, alpha=0.6, **kwargs)
    ax = kwargs.get('ax', plt.gca())
    mu, sigma = data.mean(), data.std()
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), label=fr'$\mathcal{{N}}({mu:.2f},\,{sigma:.2f}^2)$')
    ax.legend()
    if title:
        ax.set_title(title)
    print(f'Shapiro test (null=normal): p value = {stats.shapiro(data)[1]:.4f}')


def plot_normal(title='Normal Distribution', **kwargs):
    exp_norm = stats.norm.rvs(loc=0, scale=1, size=10000)
    compare_with_normal(exp_norm, title, **kwargs)


def tsplot(y, lags=30, figsize=(10, 8), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    # with plt.style.context(style):
    fig = plt.figure(figsize=figsize)
    # mpl.rcParams['font.family'] = 'Ubuntu Mono'
    layout = (3, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))
    qq_ax = plt.subplot2grid(layout, (2, 0))
    pp_ax = plt.subplot2grid(layout, (2, 1))

    y.plot(ax=ts_ax)
    ts_ax.set_title('Time Series Analysis Plots')
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
    sm.qqplot(y, line='s', ax=qq_ax)
    qq_ax.set_title('QQ Plot')
    scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

    plt.tight_layout()
    return fig


def plot_performance_quarterly(data_in, **kwargs):
    if not isinstance(data_in, pd.DataFrame):
        data_in = pd.DataFrame(data_in)
    year_quarters = []
    index = np.unique([f'{x}/{y:02d}' for x, y in zip(data_in.index.year, data_in.index.quarter)])
    for i in [2010, 2011, 2012]:
        x = data_in[data_in.index.year == i]
        year_quarters.append(x.groupby([x.index.quarter.rename('quarter')]).mean().mean(axis=1))

    plt.plot(index, pd.concat(year_quarters), **kwargs)
    plt.xticks(index, rotation=45)
    return year_quarters


def compare_industry_players(pair, corr, industry, gspc, securities_ds):
    names = [securities_ds.get_company_name(x) for x in pair]
    print(f'Correlate {pair[0]} and {pair[1]}:')
    print(f'Pearson\'s r = {corr:.2f} (without preprocessing: '
          f'{math_utils.correlation(*industry.loc[:, pair].T.values):.2f})')
    # ax = price_resids.loc[:, pair].plot(figsize=(14, 4), title=f'{names[0]} vs. {names[1]}')
    # ax.set_ylabel('Box-Cox of Open-to-close')
    ax = industry.loc[:, pair].plot(figsize=(14, 4), title=f'{names[0]} vs. {names[1]}')
    ax.plot(industry.mean(axis=1), '--', label='Energy Industry Mean', alpha=0.5)
    ax.plot(gspc[industry.index] / gspc.max() * industry.loc[:, pair].max().max(), '--', label='S&P 500 Index', alpha=0.5)
    ax.legend()
    ax.set_ylabel('Daily Opening Stock Price')
