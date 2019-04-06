from collections.abc import Iterable
import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr
from scipy.special import gamma, psi
from scipy import ndimage


# --------- Correlations ----------------------------------------------------- #
# alias "Pearson product-moment correlation coefficient" (PPMCC)
def correlation(x, y, p=False):
    if p:
        return pearsonr(x, y)
    # return np.corrcoef(x, y)[0, 1]
    return pearsonr(x, y)[0]


# Calc p manually: http://vassarstats.net/rsig.html
def calc_t(r, df):
    return round(r / np.sqrt((1-r**2)/df), 2)


def calc_r(t, df):
    return round(np.sqrt(1-(1/((t**2/df) + 1))), 2)


# statistics for p = 0.1, p = 0.05, p = 0.01 ( p = alpha / 2 )
# http://snobear.colorado.edu/Markw//IntroHydro/12/statistics/testchart.pdf
t_critical_values = pd.Series([1.65, 1.96, 2.59], index=[0.1, 0.05, 0.01])
r_critical_values = t_critical_values.apply(lambda x: calc_r(x, 500))


# TODO: (see docs)


# --------- Mutual Information & Entropy ------------------------------------- #
'''
Source: https://gist.github.com/GaelVaroquaux/ead9898bd3c973c40429
'''

EPS = np.finfo(float).eps


def nearest_distances(X, k=1):
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X)
    d, _ = knn.kneighbors(X)  # the first nearest neighbor is itself
    return d[:, -1]  # returns the distance to the kth nearest neighbor


def entropy(X, k=1):
    # Distance to kth nearest neighbor
    r = nearest_distances(X, k)  # squared distances
    n, d = X.shape
    volume_unit_ball = (np.pi**(.5*d)) / gamma(.5*d + 1)
    return (d*np.mean(np.log(r + np.finfo(X.dtype).eps))
            + np.log(volume_unit_ball) + psi(n) - psi(k))


def mutual_information_2d(x, y, sigma=1, normalized=False):
    bins = (256, 256)
    jh = np.histogram2d(x, y, bins=bins)[0]
    # smooth the jh with a gaussian filter of given sigma
    ndimage.gaussian_filter(jh, sigma=sigma, mode='constant', output=jh)

    # compute marginal histograms
    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

    if normalized:
        mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2)))
              / np.sum(jh * np.log(jh))) - 1
    else:
        mi = (np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1))
              - np.sum(s2 * np.log(s2)))
    return mi / np.log(2)  # Transform into bits


# https://stackoverflow.com/a/20505476/4816930

def shan_entropy(c):
    if isinstance(c, pd.Series):
        c = c.values
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized * np.log2(abs(c_normalized)))
    return H


def calc_mi_v1(X, Y, bins):
    c_XY = np.histogram2d(X, Y, bins)[0]
    c_X = np.histogram(X, bins)[0]
    c_Y = np.histogram(Y, bins)[0]
    H_X = shan_entropy(c_X)
    H_Y = shan_entropy(c_Y)
    H_XY = shan_entropy(c_XY)
    return H_X + H_Y - H_XY


def calc_mi_v2(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi / np.log(2)


def mutual_information(x, y, **kwargs):
    # return calc_mi_v1(x, y, **kwargs)
    # return calc_mi_v2(x, y, **kwargs)
    return mutual_information_2d(x, y, **kwargs)


def auto_mutual_information(x, lag=5):
    if isinstance(lag, int):
        lag = [lag]
    assert isinstance(lag, Iterable)
    return np.array([mutual_information(x[:-i] if i != 0 else x, x[i:]) for i in lag])


# According to Dionisio 2004
def normalized_mutual_information(x, y, **kwargs):
    # adjusted_mutual_info_score?
    mi = mutual_information(x, y, **kwargs)
    return np.sqrt(1-np.exp(-2*mi))


# m represents the length of compared run of data, r specifies a filtering level
# Wikipedia implementation is very inefficient (factor about ~15)
# Src: tsfresh.feature_extraction.feature_calculators.approximate_entropy
# Alternative: nolds.sampen (generates different results?)
def approximate_entropy(x, m=2, r=3):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)

    N = x.size
    r *= np.std(x)
    if r < 0:
        raise ValueError("Parameter r must be positive.")
    if N <= m+1:
        return 0

    def _phi(m):
        x_re = np.array([x[i:i+m] for i in range(N - m + 1)])
        C = np.sum(np.max(np.abs(x_re[:, np.newaxis] - x_re[np.newaxis, :]),
                          axis=2) <= r, axis=0) / (N-m+1)
        return np.sum(np.log(C)) / (N - m + 1.0)

    return np.abs(_phi(m) - _phi(m + 1))


def auto_correlation(x, lag=5):
    if isinstance(lag, int):
        lag = [lag]
    assert isinstance(lag, Iterable)
    return np.array([np.corrcoef(x[:-i], x[i:])[0, 1] if i != 0 else 1 for i in lag])


def cross_correlation(x, y):
    # return scipy.signal.correlate(x, y, mode='valid')
    return np.correlate(x, y, mode='valid')[0]


# Src: https://www.researchgate.net/post/How_can_one_calculate_normalized_cross_correlation_between_two_arrays
def normalized_cross_correlation(x, y):
    # return np.corrcoef(X, Y)[0, 1]
    N = len(x)
    x_mean, x_var = np.mean(x), np.var(x)
    y_mean, y_var = np.mean(y), np.var(y)
    return (1/N) * sum([(x[n] - x_mean) * (y[n] - y_mean) for n in range(N)]) / \
        np.sqrt(x_var * y_var)


def abs_values(pct, start=100):
    y = pct + 1
    y.iloc[0] = start
    return y.cumprod()
