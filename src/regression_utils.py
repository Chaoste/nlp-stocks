import itertools
from pylab import rcParams

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.stats.api as sms
import scipy.stats as scs
from arch import unitroot


def inspect_seasonality(ts, freq=252):
    rcParams['figure.figsize'] = 12, 8
    res = smt.seasonal_decompose(ts, model='additive', freq=freq)
    return res.plot()


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


def get_best_garch(TS):
    best_aic = np.inf
    best_order = None
    best_mdl = None

    pq_rng = range(5)  # [0,1,2,3,4]
    d_rng = range(2)  # [0,1]
    for i, d, j in itertools.product(pq_rng, d_rng, pq_rng):
        try:
            tmp_mdl = smt.ARIMA(TS, order=(i, d, j)).fit(
                method='mle', trend='nc'
            )
            tmp_aic = tmp_mdl.aic
            if tmp_aic < best_aic:
                best_aic = tmp_aic
                best_order = (i, d, j)
                best_mdl = tmp_mdl
        except BaseException:
            continue
    print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))
    return best_aic, best_order, best_mdl


def get_best_arima(TS):
    best_aic = np.inf
    best_order = None
    best_mdl = None

    pq_rng = range(5)  # [0,1,2,3,4]
    d_rng = range(2)  # [0,1]
    for i, d, j in itertools.product(pq_rng, d_rng, pq_rng):
        try:
            tmp_mdl = smt.ARIMA(TS, order=(i, d, j)).fit(
                method='mle', trend='nc'
            )
            tmp_aic = tmp_mdl.aic
            if tmp_aic < best_aic:
                best_aic = tmp_aic
                best_order = (i, d, j)
                best_mdl = tmp_mdl
        except BaseException:
            continue
    print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))
    return best_aic, best_order, best_mdl


def inspect_properties(ts):
    print("Test for Unit Root:")
    # Exp: For instance a ‘shock’ dies away with stationarity but is persistent if non stationary.
    # statsmodels modules cuts the p value at 0.1 (has the same statistic as unitroot module)
    # print(f'Augmented Dickey-Fuller (null = I(1)): p-value = {smt.adfuller(ts)[1]:.4f}')
    print(f'>Augmented Dickey-Fuller (null = I(1)): p value = {unitroot.ADF(ts).pvalue:.4f}')
    # print(f'KPSS (null = I(0)): p-value = {smt.kpss(ts, regression="c")[1]:.4f}')
    print(f'>KPSS (null = I(0)): p value = {unitroot.KPSS(ts).pvalue:.4f}')  # trend 'nc'/'c'/'ct'
    print(f'>Phillips-Perron (null = I(1)): p value = {unitroot.PhillipsPerron(ts).pvalue:.4f}')

    print("\nTest for Autocorrelation:")
    # 0 = pos auto correlated, 1.5-2.5 = no correlation (rule of thumb), 4 = neg auto correlated
    # https://web.stanford.edu/~clint/bench/dwcrit.htm For 700 samples: dL and dU around 1.8
    print(f'>Durbin-Watson (null(2) = no autocorr., lag 1): statistic = {sms.durbin_watson(ts):.4f}')
    print(f">Ljung-Box-Q (null = no autocorr., lag 1): p value = {sms.acorr_ljungbox(ts)[1][0]:.2f}")
    # Requires statsmodel Object e.g. from OLS(): sms.acorr_breusch_godfrey(ts, nlags = 2)

    print("\nTest for Normal Distribution:")
    print(f'>Jarque-Bera (null = gaussian): p value = {sms.jarque_bera(ts)[1]:.4f}')
    print(f'>Shapiro-Wilk (null = gaussian): p value = {stats.shapiro(ts)[1]:.4f}')
    print(f'>D’Agostino’s K^2 (null = gaussian): p value = {stats.normaltest(ts)[1]:.4f}')
    print(f'>Anderson-Darling (null = gaussian): p value = {anderson_test(ts):.4f}')

    print("\nTest for Heteroscedastiscity:")
    print(f'>Engle\'s ARCH (null = homosc.): p value = {sms.het_arch(ts)[1]:.4f}')
    # Breusch-Pagan is sensitive to missing normality, White is less sensitive
    # print(f'Goldfeld-Quandt (null = homosc.): p value = {sms.het_goldfeldquandt(ts)[1]:.4f}')
    # print(f'Breusch-Pagan (null = homosc.): p value = {sms.het_white(ts)[1]:.4f}')
    # print(f'White (null = homosc.): p value = {sms.het_white(ts)[1]:.4f}')
    # print(sms.het_white(ts.shift(lag)[lag:], sm.add_constant(ts[lag:], 1)))
    # print(stats.levene(ts.shift(lag)[lag:], ts[lag:])=
    pass


def anderson_test(ts):
    anderson = stats.anderson(ts)
    n_levels = len(anderson.critical_values)
    if anderson.statistic < anderson.critical_values[0]:
        return 1
    if anderson.statistic > anderson.critical_values[-1] * 1.5:
        return 0
    return anderson.significance_level[np.arange(n_levels)[
        anderson.critical_values <= anderson.statistic].argmax()]


def compare_distributions(x, y):
    print("\nTest for Distributions Equality:")
    print(f'>Kolmogorov-Smirnov (null = equal): p value = {stats.ks_2samp(x, y)[1]:.4f}')
    print(f'>Kruskal-Wallis (null = equal): p value = {stats.kruskal(x, y)[1]:.4f}')
    print(f'>Mann-Whitney (null = equal): p value = {stats.mannwhitneyu(x, y)[1]:.4f}')


# TODO: Tests on correlations:
# https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/?unapproved=477258&moderation-hash=c32b4646e3a75b228936bf39369d2155#comment-477258
