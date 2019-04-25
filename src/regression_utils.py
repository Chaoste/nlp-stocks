import itertools
from pylab import rcParams
from tqdm import tqdm_notebook as tqdm

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.stats.api as sms
# import statsmodels.formula.api as smf
# import statsmodels.stats.diagnostic as smsd
from sklearn.metrics import mean_squared_error as mse
from arch import unitroot, arch_model
from arch.utility.testing import WaldTestStatistic

import src.plot_utils as plot
import src.math_utils as math_utils


# ------------------ Preconditions ------------------------------------------- #


def investigate(ts, debug=False, verbose=False, exog=None):
    unit = has_unit_root(ts, verbose)
    auto = is_autocorrelated(ts, verbose)
    normal = is_normal_distributed(ts, verbose)
    hetero = is_heteroscedastic(ts, exog, verbose)
    breaks = has_structural_breaks(ts, verbose)
    outliers = len(get_outliers(ts, verbose))
    if debug:
        print(f"Does it have a unit root? {unit:.2f}")
        print(f"Is it auto-correlated? {auto:.2f}")
        print(f"Is it gaussian like distributed? {normal:.2f}")
        print(f"Is it heteroscedastic? {hetero:.2f}")
        print(f"Does it have structural breaks? {breaks:.2f}")
        print(f"Amount of outliers (assuming IQR of normality): {outliers} (expected: {int(len(ts) * 0.007)})")
    return pd.Series([unit, auto, normal, hetero, breaks, outliers],
                     index=['unit_root', 'autocorr', 'normal_dist', 'heteroscedastic', 'breaks', 'outliers'])


def has_unit_root(ts, verbose=False):
    adf = unitroot.ADF(ts).pvalue > 0.05
    pp = unitroot.PhillipsPerron(ts).pvalue > 0.05
    kpss = unitroot.KPSS(ts).pvalue <= 0.05
    if verbose:
        print("Test for Unit Root:")
        # Exp: For instance a ‘shock’ dies away with stationarity but persists if non stationary.
        # statsmodels modules cuts the p value at 0.1 (has the same statistic as unitroot module)
        # print(f'Augmented Dickey-Fuller (null = I(1)): p-value = {smt.adfuller(ts)[1]:.4f}')
        print(f'>Augmented Dickey-Fuller (null = I(1)): p value = {unitroot.ADF(ts).pvalue:.4f}')
        # print(f'KPSS (null = I(0)): p-value = {smt.kpss(ts, regression="c")[1]:.4f}')
        print(f'>KPSS (null = I(0)): p value = {unitroot.KPSS(ts).pvalue:.4f}')  # tr='nc'/'c'/'ct'
        print(f'>Phillips-Perron (null = I(1)): p value = {unitroot.PhillipsPerron(ts).pvalue:.4f}')
    return np.sum([adf, pp, kpss]) / 3.0


def is_autocorrelated(ts, verbose=False):
    # Assume no correlation for 1.8-2.2 (https://web.stanford.edu/~clint/bench/dw05d.htm)
    dw = abs(sms.durbin_watson(ts) - 2) > 0.2
    alb = sms.acorr_ljungbox(ts)[1][0] <= 0.05
    if verbose:
        print("\nTest for Autocorrelation:")
        # 0 = pos auto correlated, 1.5-2.5 = no correlation (rule of thumb), 4 = neg auto correlated
        # https://web.stanford.edu/~clint/bench/dwcrit.htm For 700 samples: dL and dU around 1.8
        print(f">Durbin-Watson (null(2) = no autocorr., lag 1): statistic"
              f" = {sms.durbin_watson(ts):.4f}")
        print(f">Ljung-Box-Q (null = no autocorr., lag 1): p value"
              f" = {sms.acorr_ljungbox(ts)[1][0]:.2f}")
        # Requires statsmodel Object e.g. from OLS(): sms.acorr_breusch_godfrey(ts, nlags = 2)
    return np.sum([dw, alb]) / 2.0


def is_normal_distributed(ts, verbose=False):
    jb = sms.jarque_bera(ts)[1] > 0.05
    sw = stats.shapiro(ts)[1] > 0.05
    dk2 = stats.normaltest(ts)[1] > 0.05
    ad = anderson_test(ts) > 0.05
    if verbose:
        print("\nTest for Normal Distribution:")
        print(f'>Jarque-Bera (null = gaussian): p value = {sms.jarque_bera(ts)[1]:.4f}')
        print(f'>Shapiro-Wilk (null = gaussian): p value = {stats.shapiro(ts)[1]:.4f}')
        print(f'>D’Agostino’s K^2 (null = gaussian): p value = {stats.normaltest(ts)[1]:.4f}')
        print(f'>Anderson-Darling (null = gaussian): p value = {anderson_test(ts):.4f}')
    return np.sum([jb, sw, dk2, ad]) / 4.0


def get_auto_lags(ts):
    nobs = ts.shape[0]
    lags = int(np.ceil(12. * np.power(nobs / 100., 1 / 4.)))
    lags = max(min(ts.shape[0] // 2 - 1, lags), 1)
    lag, lead = smt.lagmat(ts, lags, 'both', 'sep', False)
    lag = sm.add_constant(lag)
    return lead, lag


def is_heteroscedastic(orig_ts, exog=None, verbose=False):
    if exog is None:
        ts, exog = get_auto_lags(orig_ts)
    else:
        ts, exog = orig_ts, sm.add_constant(exog)
    # ea = sms.het_arch(orig_ts)[1] <= 0.05

    wh = sms.het_white(ts, exog)[1] <= 0.05
    bp = sms.het_breuschpagan(ts, exog)[1] <= 0.05

    # Requires another independent variable responsible for the changes in variance
    # gf = sms.het_goldfeldquandt(ts, exog)[1] <= 0.05

    # Not to be used on time series
    # le = levene_test(orig_ts) <= 0.05
    # fl = fligner_test(orig_ts) <= 0.05
    # wa = wald_test(orig_ts) <= 0.05

    if verbose:
        print("\nTest for Heteroscedastiscity:")
        print(f'>Engle\'s ARCH (null = homosc.): p value = {sms.het_arch(orig_ts)[1]:.4f}')
        # Breusch-Pagan is sensitive to missing normality, White is less sensitive
        print(f'>White (null = homosc.): p value = {sms.het_white(ts, exog)[1]:.4f}')
        print(f'>Breusch-Pagan (null = homosc.): p value = {sms.het_breuschpagan(ts, exog)[1]:.4f}')
        print(f'>Goldfeld-Quandt (null = homosc.): p value = {sms.het_goldfeldquandt(ts, exog)[1]:.4f}')
        # Levene (Brown-Forsythe), Fligner is suitable for violation of normality
        print(f'>Levene alias Brown-Forsythe (null = homosc.): p value = {levene_test(orig_ts):.4f}')
        print(f'>Fligner-Killeen (null = homosc.): p value = {fligner_test(orig_ts):.4f}')
        print(f'>[DEV]Wald-Test on squares (null = homosc.): p value = {wald_test(orig_ts):.4f}')
    return np.sum([wh, bp]) / 2.0


def inspect_seasonality(ts, freq=252):
    temp = rcParams['figure.figsize']
    rcParams['figure.figsize'] = 12, 8
    res = smt.seasonal_decompose(ts, model='additive', freq=freq)
    p = res.plot()
    rcParams['figure.figsize'] = temp
    return p


def has_structural_breaks(ts, verbose=False):
    ts, exog = get_auto_lags(ts)
    res = sm.OLS(ts, exog).fit()
    cusumols = sms.breaks_cusumolsresid(res.resid)[1] <= 0.05
    if verbose:
        print("\nTest for Structural Breaks:")
        print(f'>CUSUM test on OLS residuals (null = stable coeff): p value = '
              f'{sms.breaks_cusumolsresid(res.resid)[1]:.4f}')
    return float(cusumols)


def get_outliers(x, norm=False, debug=False):
    outliers = []
    if norm:  # Normal: Outliers are outside of the confidence interval 99.3%
        # Only 0.7% data points are usually outside of these bounds
        iqr = stats.iqr(x)
        lower_bound = np.percentile(x, 25) - 1.5 * iqr
        upper_bound = np.percentile(x, 75) + 1.5 * iqr
        expected_number = len(x) * 0.007
    else:  # Student's t: Outliers are outside of the confidenve interval 99%
        # loc, scale = stats.laplace.fit(x)
        # two-sided: 99.5%=6.215, 99.9%=6.907
        # critical_value = stats.laplace.interval(0.995)
        df, loc, scale = stats.t.fit(x)
        # two-sided (df=inf): 99%=2.576, 99.5%=2.807, 99.9%=3.291
        lower_bound, upper_bound = stats.t.interval(0.995, df, loc=loc, scale=scale)
        # lower_bound = loc - critical_value * scale
        # upper_bound = loc + critical_value * scale
        expected_number = len(x) * 0.005
    for i, val in enumerate(x):
        if val < lower_bound or val > upper_bound:
            outliers.append(i)
    if debug:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        pd.Series(x).plot(ax=axes[0])
        axes[0].set_xlabel('')
        axes[0].axhline(np.percentile(x, 50), color='lightgray', linestyle='dashed')  # median
        axes[0].axhline(lower_bound, color='orange')
        axes[0].axhline(upper_bound, color='orange')

        pd.Series(x).hist(ax=axes[1], bins=100)
        axes[1].axvline(np.percentile(x, 25), color='gray', linestyle='dashed')  # Q1
        axes[1].axvline(np.percentile(x, 50), color='lightgray', linestyle='dashed')  # median
        axes[1].axvline(np.percentile(x, 75), color='gray', linestyle='dashed')  # Q3
        axes[1].axvline(lower_bound, color='orange')
        axes[1].axvline(upper_bound, color='orange')
        for idx in outliers:
            axes[1].scatter(x[idx], 10, color='gray', alpha=0.5)
        print(f'Expected outliers: {expected_number} (vs. {len(outliers)})')
    return outliers


def wald_test(ts):
    ts, exog = get_auto_lags(ts**2)
    nobs, lags = ts.shape[0], exog.shape[1]
    res = sm.OLS(ts, exog).fit()
    stat = nobs * res.rsquared
    return WaldTestStatistic(stat, df=lags, null='homo', alternative='hetero').pval


def chunks(l, n):
    return list(np.array_split(list(l), len(l) // n))


# The size of number of chunks is unclear, so we will take the mean for several sizes
def levene_test(ts):
    values = pd.Series(index=np.arange(10, 60, 10))
    for sample_size in values.index:
        values[sample_size] = stats.levene(*chunks(ts, sample_size)).pvalue
    return values.mean()


# Same as levene_test
def fligner_test(ts):
    values = pd.Series(index=np.arange(10, 60, 10))
    for sample_size in values.index:
        values[sample_size] = stats.fligner(*chunks(ts, sample_size)).pvalue
    return values.mean()


# Supposed to beat Kolmogorov-Smirnov (smsd.kstest_normal)
def anderson_test(ts):
    anderson = stats.anderson(ts)
    n_levels = len(anderson.critical_values)
    if anderson.statistic < anderson.critical_values[0]:
        return 1
    if anderson.statistic > anderson.critical_values[-1] * 1.5:
        return 0
    return anderson.significance_level[np.arange(n_levels)[
        anderson.critical_values <= anderson.statistic].argmax()] / 100


def compare_distributions(x, y):
    print("\nTest for Distributions Equality:")
    print(f'>Kolmogorov-Smirnov (null = equal): p value = {stats.ks_2samp(x, y)[1]:.4f}')
    print(f'>Kruskal-Wallis (null = equal): p value = {stats.kruskal(x, y)[1]:.4f}')
    print(f'>Mann-Whitney (null = equal): p value = {stats.mannwhitneyu(x, y)[1]:.4f}')


# ------------------ Modelling ----------------------------------------------- #

# Available models: GARCH, ARCH, EGARCH, FIARCH and HARCH
def get_best_garch(ts, r=range(6), s=range(6), debug=False, model='GARCH'):
    aic_values = pd.Series(index=pd.MultiIndex.from_product([r, s]))

    for i, j in itertools.product(r, s):
        try:
            tmp_mdl = arch_model(ts, mean='Zero', vol=model, p=i, q=j).fit(
                update_freq=5, disp='off')
            aic_values[i, j] = tmp_mdl.aic
        except BaseException as e:
            # e.g. "ValueError: The computed initial AR coefficients are not stationary" (p==q)
            continue
    print('Best AIC: {:.4f} (worst: {:.4f}) | params: {}, {}'.format(
        aic_values.max(), aic_values.min(), *aic_values.idxmax()))
    if debug:
        return aic_values
    return aic_values.max(), aic_values.idxmax()


def get_best_arima(ts, p=range(1, 5), d=range(1), q=range(5), exog=None, debug=False, verbose=True):
    aic_values = pd.Series(index=pd.MultiIndex.from_product([p, d, q]))
    combinations = itertools.product(p, d, q)
    for i, j, k in (tqdm(list(combinations)) if verbose else combinations):
        if i == j == k == 0:
            continue
        try:
            tmp_mdl = smt.ARIMA(ts, exog=exog, order=(i, j, k)).fit(
                method='mle', trend='nc', disp=0,
            )
            aic_values[i, j, k] = tmp_mdl.aic
        except ValueError:
            continue
            # e.g. "ValueError: The computed initial AR coefficients are not stationary" (p==q)
        # except BaseException as e:
        #     raise e
        #     continue
    if verbose:
        print('Best AIC: {:.4f} (worst: {:.4f}) | params: {}, {}, {}'.format(
            aic_values.max(), aic_values.min(), *aic_values.idxmax()))
    if debug:
        return aic_values
    return aic_values.max(), aic_values.idxmax()


def determine_model_params(ts, nlags=10, plot=False):
    pacf_vals, pconfint = smt.pacf(ts, nlags=nlags, alpha=0.05)
    pacf_vals, pconfint = pacf_vals[1:], pconfint[1:]
    acf_vals, confint = smt.acf(ts, nlags=nlags, alpha=0.05)
    acf_vals, confint = acf_vals[1:], confint[1:]
    # conf = stats.norm.ppf(1. - 0.05 / 2.) * np.sqrt(1 / len(ts))
    p, q = 0, 0
    while p < 10 and np.sign(pconfint[p, 0]) == np.sign(pconfint[p, 1]):
        p += 1
    while q < 10 and np.sign(confint[q, 0]) == np.sign(confint[q, 1]):
        q += 1
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        smt.graphics.plot_acf(ts, lags=nlags, alpha=0.05, ax=axes[0])
        axes[0].axvline(q, alpha=0.2, linestyle='dashed')
        smt.graphics.plot_pacf(ts, lags=nlags, alpha=0.05, ax=axes[1])
        axes[1].axvline(p, alpha=0.2, linestyle='dashed')
    return p, q


def model_ARMA(ts, p=None, q=None):
    if p is None or q is None:
        p, q = determine_model_params(ts)
    if p != 0 or q != 0:
        try:
            resid = smt.ARIMA(ts, order=(p, 0, q)).fit(
                method='mle', trend='nc', update_freq=5, disp=0).resid
        except ValueError as e:
            if (str(e).find('coefficients are not stationary') == -1 and
                    str(e).find('coefficients are not invertible') == -1):
                raise
            p += 1
            resid = smt.ARIMA(ts, order=(p, 0, q)).fit(
                method='mle', trend='nc', update_freq=5, disp=0).resid
    else:
        resid = ts
    return resid, p, q


def model_GARCH(ts, r=None, s=None, debug=False):
    if r is None or s is None:
        r, s = determine_model_params(ts**2)
    if r != 0 or s != 0:
        arch_fitted = arch_model(ts, mean='Zero', vol='GARCH', p=r, q=s).fit(
            update_freq=5, disp='off')
        std_resid = arch_fitted.resid / arch_fitted.conditional_volatility.values
        if debug:
            print(arch_fitted.arch_lm_test(standardized=True))
    else:
        # To bring the values into the equal scale independent of a GARCH model
        std_resid = ts / ts.std()
    return std_resid, r, s


def predict_rolling_forward(train_ts, val_ts, arima_params):
    history = list(train_ts)
    ex_sample_predictions = pd.DataFrame(
        0, columns=['value', 'std', 'confLow', 'confUp'], index=val_ts.index)
    for t in tqdm(range(len(val_ts))):
        model = smt.ARIMA(history, order=arima_params)
        model_fit = model.fit(method='mle', trend='nc', update_freq=5, disp=0)
        guess, std, confs = model_fit.forecast()
        ex_sample_predictions.iloc[t] = (guess, std[0], *confs[0])
        history.append(val_ts[t])
    return ex_sample_predictions


def predict_garch_rolling_forward(train_ts, val_ts, arima_params, garch_params, vol_model='GARCH'):
    history = list(train_ts)
    ex_sample_predictions = pd.DataFrame(
        0, columns=['value', 'variance', 'mean'], index=val_ts.index)
    for t in tqdm(range(len(val_ts))):
        model = ARIMA_GARCH(history, *arima_params, *garch_params, vol_model=vol_model)
        res = model.garch_fit
        forecast = res.forecast()
        # forecast() gives very low unreasonble results - calculate manually
        resid = pd.Series(res.resid)
        cond_vol = pd.Series(res.conditional_volatility)
        alpha_sum = sum([res.params.get(f'alpha[{i+1}]', 0) * resid.shift(i)**2 for i in range(model.r)])
        beta_sum = sum([res.params.get(f'beta[{i+1}]', 0) * cond_vol.shift(i)**2 for i in range(model.s)])
        res = model.garch_fit
        predictions = np.sqrt(res.params['omega'] + alpha_sum + beta_sum)
        predictions[:max(model.r, model.s)-1] = res.conditional_volatility[1:max(model.r, model.s)]
        ex_sample_predictions.iloc[t] = (predictions.iloc[-1], forecast.variance.iloc[-1, 0],
                                         forecast.mean.iloc[-1, 0])
        history.append(val_ts[t])
    return ex_sample_predictions


# ------------------ Predictions --------------------------------------------- #


def rms(y_true, y_pred):  # Root mean squared error
    return round(np.sqrt(mse(y_true, y_pred)), 4)


def plot_arima_predictions(train_ts, val_ts, arima_params):
    model_name = f'ARIMA{arima_params}'
    model = smt.ARIMA(train_ts, order=arima_params)
    model_fit = model.fit(method='mle', trend='nc', update_freq=5)
    in_sample_predictions = model_fit.predict()  # .fittedvalues
    ex_sample_predictions = predict_rolling_forward(train_ts, val_ts, arima_params)

    ax = train_ts.plot(label='Train', alpha=0.3,
                       title=f'Model & Rolling Forecast of {model_name}', figsize=(12, 4))
    ax.plot(train_ts.index, in_sample_predictions, label='In-Sample')
    val_ts.plot(ax=ax, label='Validation', alpha=0.3)
    ex_sample_predictions.value.plot(ax=ax, label='Ex-Sample')
    plt.xlim((train_ts.index[0], val_ts.index[-1]))
    plt.axvline(val_ts.index[0], color='black', linestyle='dashed')
    ax.legend()
    ax.set_ylabel('Link Relatives')
    ax.set_xlabel('')
    # plt.gcf().savefig('{symbol} - {model_name}.pdf')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    print('Training data:')
    print(f'> {model_name}, RMS = ', rms(train_ts, in_sample_predictions))
    print('> Persistence Model, RMS = ', rms(train_ts, train_ts.shift(1).fillna(0)))
    print('Validation data:')
    print(f'> {model_name}, RMS = ', rms(val_ts, ex_sample_predictions.value))
    print('> Persistence Model, RMS = ', rms(val_ts, val_ts.shift(1).fillna(0)))

    axes[0].plot(train_ts - train_ts.shift(1), alpha=0.4, label='Persistence Model Error')
    (train_ts - in_sample_predictions).plot(ax=axes[0], alpha=0.8, label='ARIMA Error')
    axes[0].legend()
    axes[0].set_title('Model Errors vs. Baseline')

    axes[1].hist(train_ts, alpha=0.4, bins=100, color='gray', label='Original')
    plot.compare_with_normal(model_fit.resid, title='Distribution of ARIMA Errors', ax=axes[1])


class ARIMA_GARCH():
    def __init__(self, data, p=1, d=0, q=1, r=3, s=3, vol_model='HARCH'):
        self.arima = smt.ARIMA(data, order=(p, d, q))
        self.arima_fit = self.arima.fit(method='mle', trend='nc', update_freq=5)
        # Use ARIMAs residuals
        self.garch = arch_model(self.arima_fit.resid, mean='Zero', vol=vol_model, p=r, q=s)
        self.garch_fit = self.garch.fit(update_freq=5, disp='off')
        # https://arch.readthedocs.io/en/latest/univariate/introduction.html
        self.in_sample_predictions = self.garch_fit.conditional_volatility
        self.p, self.d, self.q, self.r, self.s = p, d, q, r, s
        # fig = res.plot(annualize='D')
        # sms.het_arch(res.resid / res.conditional_volatility)
        # am = arch_model(train, p=1, o=0, q=1, dist='StudentsT', vol='EGARCH')

    def predict_volatility(self):
        yhat = self.garch_fit.forecast()
        return yhat.variance.iloc[-1][0]

    def __str__(self):
        return f'{self.arima.__class__.__name__}({self.p}, {self.d}, {self.q})-' \
               f'{self.garch.garch.volatility.__class__.__name__}({self.r}, {self.s})'


def plot_garch_predictions(train_ts, val_ts, arima_params, garch_params, vol_model='GARCH'):
    model_name = f'{vol_model.upper()}{arima_params + garch_params}'
    model = ARIMA_GARCH(train_ts, *arima_params, *garch_params, vol_model=vol_model)
    _final_model = ARIMA_GARCH(
        pd.concat([train_ts, val_ts]), *arima_params, *garch_params, vol_model=vol_model)
    in_sample_predictions = model.in_sample_predictions
    ex_sample_predictions = predict_garch_rolling_forward(
        train_ts, val_ts, arima_params, garch_params, vol_model)

    # There is no real volatility - this is an approximations we're using for comparison
    window = max(*garch_params, 2)
    train_volatility = model.arima_fit.resid.rolling(window).std(ddof=0)
    # Due to rolling window fill nones with last actual value
    train_volatility[:window-1] = train_volatility[window-1]
    val_volatility = _final_model.arima_fit.resid[1-len(val_ts)-window:]\
        .rolling(window).std(ddof=0).dropna()

    ax = train_volatility.plot(label='Train', alpha=0.7, figsize=(12, 4),
                               title=f'Model & Rolling Forecast of {model_name}')
    ax.plot(train_ts.index, in_sample_predictions, label='In-Sample', alpha=0.6)
    val_volatility.plot(ax=ax, label='Validation', alpha=0.7)
    ex_sample_predictions.value.plot(ax=ax, label='Ex-Sample', alpha=0.6)
    plt.xlim((train_ts.index[0], val_ts.index[-1]))
    plt.axvline(val_ts.index[0], color='black', linestyle='dashed')
    ax.legend()
    ax.set_ylabel('Volatility of Link Relatives')
    ax.set_xlabel('')
    # plt.gcf().savefig(f'gspc - {model}.pdf')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    _final_model.arima_fit.resid.plot(ax=axes[1])
    axes[1].set_title('ARIMA - Residuals')

    print('Training data:')
    print(f'> {model_name}, RMS = ', rms(train_volatility, in_sample_predictions))
    print('> Persistence Model, RMS = ', rms(train_volatility, train_volatility.shift(1).fillna(0)))
    print('Validation data:')
    print(f'> {model_name}, RMS = ', rms(val_volatility, ex_sample_predictions.value))
    print('> Persistence Model, RMS = ', rms(val_volatility, val_volatility.shift(1).fillna(0)))

    (train_volatility - in_sample_predictions).plot(ax=axes[0], label='GARCH Error')
    axes[0].plot(train_volatility - train_volatility.shift(1), alpha=0.7, label='Persistence Model Error')
    axes[0].legend()
    axes[0].set_title('Model Errors')
    return ex_sample_predictions


# ------------------ Correlations -------------------------------------------- #
# TODO: Tests on correlations:
# https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/?unapproved=477258&moderation-hash=c32b4646e3a75b228936bf39369d2155#comment-477258


def calc_correlations(comp_symbols, comp_prices):
    correlations = pd.DataFrame(np.nan, index=comp_symbols, columns=comp_symbols)
    for a, b in tqdm(itertools.product(comp_symbols, comp_symbols), total=len(comp_symbols)**2):
        if a <= b:
            continue
        correlations.loc[a, b] = math_utils.correlation(comp_prices.loc[:, a], comp_prices.loc[:, b])
        correlations.loc[b, a] = correlations.loc[a, b]
    return correlations


def calc_all_correlations(comp_symbols, comp_resids, comp_prices, comp_orig_returns, comp_orig_prices):
    correlations_orig_prices = calc_correlations(comp_symbols, comp_orig_prices)
    correlations_orig_returns = calc_correlations(comp_symbols, comp_orig_returns)
    correlations_normed = calc_correlations(comp_symbols, comp_prices)
    correlations_resid = calc_correlations(comp_symbols, comp_resids)
    all_correlations = pd.concat([
        correlations_orig_prices.stack().rename('Price'),
        correlations_orig_returns.stack().rename('Return'),
        correlations_normed.stack().rename('Normalized'),
        correlations_resid.stack().rename('Residuals'),
    ], axis=1)
    all_correlations = all_correlations[[a < b for a, b in all_correlations.index]]
    return all_correlations


def plot_correlations(all_correlations):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.violinplot(all_correlations.T)
    ax.grid(True, color='lightgray')
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(all_correlations.columns.values)
    # ax.set_title('Correlations');
    return fig
