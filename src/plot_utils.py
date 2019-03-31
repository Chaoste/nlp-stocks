import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


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
