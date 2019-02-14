import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr
from scipy.special import gamma, psi
from scipy import ndimage


# --------- Correlations ----------------------------------------------------- #
# alias "Pearson product-moment correlation coefficient" (PPMCC)
def correlation(x, y):
    # return np.corrcoef(x, y)[0, 1]
    return pearsonr(x, y)[0]

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
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized * np.log2(c_normalized))
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


def calc_mi(x, y, **kwargs):
    # return calc_mi_v1(x, y, **kwargs)
    # return calc_mi_v2(x, y, **kwargs)
    return mutual_information_2d(x, y, **kwargs)
