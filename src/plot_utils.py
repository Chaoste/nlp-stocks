import os
import time
import locale
import datetime
import calendar

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import scipy.stats as scs
import statsmodels.tsa.api as smt
import matplotlib
import matplotlib.pyplot as plt

import src.math_utils as math_utils

# Use dots for thousand steps and comma for decimal digits
locale.setlocale(locale.LC_ALL, 'en_US.utf-8')

primary = '#037d95'  # blue green
secondary = '#ffa823'  # orange yellow
ternary = '#c8116b'  # red violet
colors = (primary, secondary, ternary)
# http://www.somersault1824.com/tips-for-designing-scientific-figures-for-color-blind-readers/


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


def compare_with_t(data, norm=True, title=None, **kwargs):
    pd.DataFrame(data).hist(bins=100, density=True, alpha=0.6, **kwargs)
    ax = kwargs.get('ax', plt.gca())
    df, loc, scale = stats.t.fit(data)
    x = np.linspace(np.min(data), np.max(data), 100)
    ax.plot(x, stats.t.pdf(x, df, loc, scale), label='Student\'s $t$')
    if norm:
        mu, sigma = data.mean(), data.std()
        ax.plot(x, stats.norm.pdf(x, mu, sigma), label=fr'$\mathcal{{N}}({mu:.2f},\,{sigma:.2f}^2)$')
    ax.legend()
    if title:
        ax.set_title(title)
    print(f'KS test (null=equal): p value = {stats.kstest(data, "t", args=(df, loc, scale))[1]:.2f}')


def compare(data, title=None):
    pd.DataFrame(data).hist(bins=100, density=True, alpha=0.6)
    ax = plt.gca()
    params = stats.laplace.fit(data)
    x = np.linspace(np.min(data), np.max(data), 100)
    ax.plot(x, stats.laplace.pdf(x, *params), label='Laplace')
    mu, sigma = data.mean(), data.std()
    ax.plot(x, stats.norm.pdf(x, mu, sigma), label=fr'$\mathcal{{N}}({mu:.2f},\,{sigma:.2f}^2)$')
    df, loc, scale = stats.t.fit(data)
    x = np.linspace(np.min(data), np.max(data), 100)
    ax.plot(x, stats.t.pdf(x, df, loc, scale), label='Student\'s $t$')
    ax.legend()
    return ax


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
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.05)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.05)
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


def scatter_regression(x, y):
    gradient, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    mn = np.min(x)
    mx = np.max(x)
    x1 = np.linspace(mn, mx, 500)
    y1 = gradient*x1+intercept
    plt.plot(x, y, 'o', alpha=0.4)
    plt.plot(x1, y1, '-')


def compare_industry_players(pair, corr, industry, industry_orig, gspc, securities_ds):
    names = [securities_ds.get_company_name(x) for x in pair]
    print(f'Correlate {pair[0]} and {pair[1]}:')
    print(f'Pearson\'s r = {corr:.2f} (without preprocessing: '
          f'{math_utils.correlation(*industry_orig.loc[:, pair].T.values):.2f})')
    # ax = price_resids.loc[:, pair].plot(figsize=(14, 4), title=f'{names[0]} vs. {names[1]}')
    # ax.set_ylabel('Box-Cox of Open-to-close')
    ax = industry_orig.loc[:, pair].plot(figsize=(14, 4), title=f'{names[0]} vs. {names[1]}')
    # ax.plot(industry_orig.mean(axis=1), '--', label='Energy Industry Mean', alpha=0.5)
    ax.plot(gspc[industry_orig.index] / gspc.max() * industry_orig.loc[:, pair].max().max(), '--', label='S&P 500 Index', alpha=0.5)
    ax.plot(math_utils.abs_values(industry[pair[0]], industry_orig[pair[0]][0]), color='#1f77b4', ls='--', label=f'{pair[0]} [norm]')
    ax.plot(math_utils.abs_values(industry[pair[1]], industry_orig[pair[1]][0]), color='#ff7f0e', ls='--', label=f'{pair[1]} [norm]')
    ax.legend()
    ax.set_ylabel('Daily Opening Stock Price')


def plot_acf_pacf(x, sym, securities, lags=10):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    smt.graphics.plot_acf(x, lags=lags, ax=axes[0], alpha=None, color=primary)
    acf_x, confint = smt.acf(x, nlags=lags, alpha=0.05)
    confint -= np.array([acf_x, acf_x]).T
    confint = np.concatenate([confint, confint[-1:, :]])
    axes[0].fill_between(np.arange(lags+2), confint[:, 0], confint[:, 1], alpha=.25, color=primary)
    axes[0].set_xlim((-0.2, 5.2))
    axes[0].set_ylim((-0.2, 0.4))
    axes[0].set_ylabel('ACF')
    axes[0].set_xlabel('lag')

    smt.graphics.plot_pacf(x, lags=lags, ax=axes[1], alpha=None, color=primary)
    pacf_x, confint = smt.pacf(x, nlags=lags, alpha=0.05)
    confint -= np.array([pacf_x, pacf_x]).T
    confint = np.concatenate([confint, confint[-1:, :]])
    axes[1].fill_between(np.arange(lags+2), confint[:, 0], confint[:, 1], alpha=.25, color=primary)
    axes[1].set_xlim((-0.2, 5.2))
    axes[1].set_ylim((-0.2, 0.4))
    axes[1].set_ylabel('PACF')
    axes[1].set_xlabel('lag')

    fig.suptitle(f'{securities.get_company_name(sym)} ({sym})')
    fig.tight_layout()
    fig.subplots_adjust(top=0.8)
    return fig, axes


def get_month(i):
    return datetime.date(2000, int(i), 1).strftime('%B')


def get_weekday(i):
    return calendar.day_name[int(i)]


def boxplot_monthly(r, ax=None):
    monthly_returns = r.groupby([r.index.year.rename('year'), r.index.month.rename('month')]).mean()
    monthly_returns = pd.DataFrame(monthly_returns.reset_index().values, columns=('year', 'month', 'return'))
    ax = monthly_returns.boxplot(column='return', by='month', ax=ax)
    ax.set_title('')
    plt.xticks(monthly_returns.iloc[:12].month, [get_month(x) for x in monthly_returns.iloc[:12].month], rotation=45)
    plt.tick_params(axis='both', which='major')


def _to_rgb(cmap, step, as_string=True):
    r, g, b, _ = cmap(step)
    if as_string:
        return f'rgb({int(256*r)}, {int(256*g)}, {int(256*b)})'
    return np.array((int(256*r), int(256*g), int(256*b)))


def get_colors(ents, as_string=True):
    cmap_name = 'Set3' if len(ents) > 8 else 'Pastel2'
    steps = np.linspace(0, 1, 12 if len(ents) > 8 else 8)
    cmap = matplotlib.cm.get_cmap(cmap_name)
    return dict([(e, _to_rgb(cmap, steps[i], as_string)) for i, e in enumerate(ents)])


# --- Seasonality ---- #


def plot_weekdays(r, store_path=False, comp=False):
    if comp:
        daily_returns = r.groupby([r.index.weekday.rename('weekday'), r.sym]).mean()
        daily_returns = pd.DataFrame(daily_returns.reset_index().values, columns=('weekday', 'sym', 'return'))
    else:
        daily_returns = r.rename('return').reset_index().rename_axis({'date': 'weekday'}, axis=1)
        daily_returns.weekday = daily_returns.weekday.dt.weekday
    fig, ax = plt.subplots(figsize=(7, 3))
    ax = daily_returns.boxplot(column='return', by='weekday', ax=ax)
    fig.suptitle('Return grouped by weekday')
    ax.set_title('')
    ax.set_xlabel('')
    # labels = [item.get_text() for item in ax.get_xticklabels()]
    plt.xticks(daily_returns.weekday.unique() + 1, [get_weekday(x) for x in daily_returns.weekday.unique()])
    plt.tick_params(axis='both', which='major')
    if store_path is not False:
        fig.savefig(store_path)
    return daily_returns


def plot_weeks(r, store_path=False, comp=False):
    if comp:
        weekly_returns = r.groupby([r.index.week.rename('week'), r.sym]).mean()
        weekly_returns = pd.DataFrame(weekly_returns.reset_index().values, columns=('week', 'sym', 'return'))
    else:
        weekly_returns = r.groupby([r.index.year.rename('year'), r.index.week.rename('week')]).mean()
        weekly_returns = pd.DataFrame(weekly_returns.reset_index().values, columns=('year', 'week', 'return'))
    fig, ax = plt.subplots(figsize=(7, 3))
    ax = weekly_returns.boxplot(column='return', by='week', ax=ax)

    fig.suptitle('Return grouped by week')
    ax.set_title('')
    # labels = [item.get_text() for item in ax.get_xticklabels()]
    plt.xticks(weekly_returns.week.unique()[::2], [int(x) for x in weekly_returns.week.unique()[::2]])
    plt.tick_params(axis='both', which='major')
    if store_path is not False:
        fig.savefig(store_path)
    return weekly_returns


def plot_months(r, store_path=False, comp=False):
    if comp:
        monthly_returns = r.groupby([r.index.month.rename('month'), r.sym]).mean()
        monthly_returns = pd.DataFrame(monthly_returns.reset_index().values, columns=('month', 'sym', 'return'))
    else:
        monthly_returns = r.groupby([r.index.year.rename('year'), r.index.month.rename('month')]).mean()
        monthly_returns = pd.DataFrame(monthly_returns.reset_index().values, columns=('year', 'month', 'return'))
    fig, ax = plt.subplots(figsize=(9, 4))
    # ax = monthly_returns.boxplot(column='return', by='month', ax=ax)
    violin_parts = ax.violinplot(pd.DataFrame(
        [x[1].reset_index(drop=True).rename(x[0])
         for x in monthly_returns['return'].groupby(monthly_returns.month)]))
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('Model Residuals')
    ax.yaxis.grid()  # horizontal lines
    for pc in violin_parts['bodies']:
        pc.set_color(primary)
    for partname in ('cbars', 'cmins', 'cmaxes'):
        vp = violin_parts[partname]
        vp.set_color(primary)
    # labels = [item.get_text() for item in ax.get_xticklabels()]
    plt.xticks(monthly_returns.month.unique(), [get_month(x) for x in monthly_returns.month.unique()], rotation=45, horizontalalignment="right")
    plt.tick_params(axis='both', which='major')

    ax.set_title('Return grouped by month')
    fig.tight_layout()

    if store_path is not False:
        fig.savefig(store_path)
    return monthly_returns


# Source
# http://abhay.harpale.net/blog/python/how-to-plot-multicolored-lines-in-matplotlib/
def find_contiguous_colors(colors):
    # finds the continuous segments of colors and returns those segments
    segs = []
    curr_seg = []
    prev_color = ''
    for c in colors:
        if c == prev_color or prev_color == '':
            curr_seg.append(c)
        else:
            segs.append(curr_seg)
            curr_seg = []
            curr_seg.append(c)
        prev_color = c
    segs.append(curr_seg)
    return segs


def plot_multicolored_lines(x, y, colors):
    segments = find_contiguous_colors(colors)
    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    start = 0
    for seg in segments:
        end = start + len(seg)
        if seg[0] == 'gray':
            if start in x:
                ax.axvline(x[start], linestyle='--', color=seg[0], alpha=0.5)
            if end-1 in x:
                ax.axvline(x[end-1], linestyle='--', color=seg[0], alpha=0.5)
            l, = ax.plot(x[start:end], y[start:end], lw=2, c=seg[0])
        elif start != 0:
            l, = ax.plot(x[start-1:end+1], y[start-1:end+1], lw=2, c=seg[0])
        else:
            l, = ax.plot(x[start:end+1], y[start:end+1], lw=2, c=seg[0])
        if seg[0] != 'gray' and end in x:
            print(f'{y[max(0, start-1)]:.2f},  {y[end]:.2f}, {1 - y[max(0, start-1)] / y[end]:.2f}')
        start = end
    ax.yaxis.grid()
    return ax

# -------------------------------------- General Plotter ----------------------------------------- #


PRESENTATION = False
SEABORN_TICK_COLOR = '#555555'
PERCENTAGE_UNIT = ['selectivity', 'percentile']  # Should be given in Range [0, 100], casing doesn't matter
TEXT_MAPPING = {
    'nodes': 'Shared Nodes',
    'edges': 'Shared Edges'
}
DEFAULT_PLOTS_DIR = "."


def create_plot(x, x_label, y1, y1_label, y2=None, y2_label=None, title='',
                label=None, y1_color='#037d95', y2_color='#ffa823', ax=None,
                y1_lim=None, y2_lim=None, log=False, bars=False, multiple_ydata=None):
    if ax is None:
        _, ax = plt.subplots()
    if PRESENTATION:
        # By default show no labels in presentation mode
        label, title = [None]*2

    x, x_label, y1, y1_label, y2, y2_label, y1_lim, y2_lim = handle_units(
        x, x_label, y1, y1_label, y2, y2_label, y1_lim, y2_lim, log)
    # FIXME:
    # assert label is None or y2_label is None, 'No twin axes with multiple line plots'
    assert y1_color and y2_color

    add_to_axes(ax, x, y1, y1_color, y1_label, y1_lim, bars)
    ax2 = None
    if y2 is not None:
        ax2 = ax.twinx()
        add_to_axes(ax2, x, y2, y2_color, y2_label, y2_lim, bars, multiple_ydata)
    prettify_axes(ax, ax2)
    prettify_labels(ax, ax2, x_label, y1_label, y2_label, y1_color, y2_color, log, bars)

    if label and not y2_label:
        ax.legend()
    if not log and not bars:
        delta = 0.01 * (x[-1] - x[0])
        ax.set_xlim(x[0] - delta, x[-1] + delta)
        if x_label == 'n_cores':
            # Hardcoded for experiment multicore to show 0 and 80:
            ax.set_xlim(x[0] - 1, x[-1] + 3)
    elif not bars:
        ax.set_xscale('log')
    if not PRESENTATION:
        ax.set_xlabel(x_label)
        ax.set_title(title)
    elif x_label[0] == '[':
        ax.set_xlabel(x_label)
    return ax


def add_to_axes(ax, x, y, color, label, limits, bars=False, multiple_ydata=None):
    if len(x) == 2 or bars:
        ax.bar(x, y, color=colors[:len(x)])
    else:
        if multiple_ydata is None:
            ax.plot(x, y, color=color, label=label)
        else:
            for (_, val), c in zip(multiple_ydata.items(), colors[1:]):
                ax.plot(x, val, color=c)

    if limits:
        if limits[1] is None:
            limits = (limits[0], ax.get_ylim()[1])
        ax.set_ylim(limits[0], limits[1])


def prettify_axes(ax, ax2):
    ax.set_facecolor('white')
    ax.grid(False)
    ax.yaxis.grid(True)
    ax.spines['left'].set_color(SEABORN_TICK_COLOR)
    ax.spines['bottom'].set_color(SEABORN_TICK_COLOR)
    if ax2:
        ax2.grid(False)
        ax2.spines['left'].set_color(ax.get_yticklines()[0].get_color())
        ax2.spines['bottom'].set_color(SEABORN_TICK_COLOR)
        ax2.spines['right'].set_color(ax2.get_yticklines()[0].get_color())


def prettify_labels(ax, ax2, x_label, y1_label, y2_label, y1_color, y2_color, log, bars=False):
    if not PRESENTATION:
        ax.set_ylabel(y1_label)
    if y1_label:
        if not PRESENTATION or y1_label[0] == '[':
            ax.set_ylabel(y1_label, color=y1_color)
        # ax.tick_params('y', color=y1_color)
    if y2_label:
        if not PRESENTATION or y2_label[0] == '[':
            ax2.set_ylabel(y2_label, color=y2_color)
        # ax2.tick_params('y', color=y2_color)

        # Align ticks of y2 and y1 and keep y2 ticks integer if it already was
        ax.set_yticks(np.linspace(ax.get_yticks()[0], ax.get_yticks()[-1], len(ax.get_yticks())))
        y2_ticks = np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax.get_yticks()))
        step_size = abs(ax2.get_yticks()[0] - ax2.get_yticks()[1])
        has_y2_integer_step_size = int(step_size) == step_size
        ax2.set_yticks(y2_ticks)
        # if PRESENTATION:
        #     ax2.ticklabel_format(axis='yaxis', style='plain', useOffset=False)

    # if PRESENTATION:
        # TODO: Do we still need this since we already use format() below
        # ax.ticklabel_format(axis='yaxis' if log else 'both', style='plain', useOffset=False)
    if y1_label and y1_label.lower() in PERCENTAGE_UNIT:
        ax.set_yticklabels([f'{float(x) * 100:,.1f}%' for x in ax.get_yticks()])
    else:
        step_size = abs(ax.get_yticks()[0] - ax.get_yticks()[1])
        has_integer_step_size = int(step_size) == step_size
        ytick_labels = [locale.format('%d' if has_integer_step_size else '%.2f', x, 1) for x in ax.get_yticks()]
        if not has_integer_step_size and all([x[-1:] == '0' for x in ytick_labels]):
            ytick_labels = [x[:-1] for x in ytick_labels]
        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels(ytick_labels)
    if y2_label and y2_label.lower() in PERCENTAGE_UNIT:
        ax2.set_yticklabels([f'{float(x) * 100:,.1f}%' for x in ax2.get_yticks()])
    elif y2_label:
        # Defined above
        # step_size = abs(ax2.get_yticks()[0] - ax2.get_yticks()[1])
        # has_integer_step_size = int(step_size) == step_size
        # %.3g to allow up to three signs
        ytick2_labels = [locale.format('%d' if has_y2_integer_step_size else '%.2f', x, 1) for x in ax2.get_yticks()]
        if not has_y2_integer_step_size and all([x[-1:] == '0' for x in ytick2_labels]):
            ytick2_labels = [x[:-1] for x in ytick2_labels]
        ax2.set_yticks(ax2.get_yticks())
        ax2.set_yticklabels(ytick2_labels)
    if x_label.lower() in PERCENTAGE_UNIT:
        xtick_labels = [f'{float(x) * 100:,.1f}%' for x in ax.get_xticks()]
        if all([x[-3:] == '.0%' for x in xtick_labels]):
            xtick_labels = [f'{x[:-3]}%' for x in xtick_labels]
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(xtick_labels)
    else:
        step_size = abs(ax.get_xticks()[0] - ax.get_xticks()[1])
        has_integer_step_size = int(step_size) == step_size
        # %.3g to allow up to three signs
        xtick_labels = [locale.format('%d' if has_integer_step_size else '%.2f', x, 1) for x in ax.get_xticks()]
        if not has_integer_step_size and all([x[-1:] == '0' for x in xtick_labels]):
            xtick_labels = [x[:-1] for x in xtick_labels]
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(xtick_labels)
    if bars:
        ax.xaxis.set_ticks([])


def handle_units(x, x_label, y1, y1_label, y2=None, y2_label=None, y1_lim=None, y2_lim=None, log=False):
    if not log and max(x) >= 1e9:
        x = [x / 1e9 for x in x]
        x_label = '[Bn]'
    elif not log and max(x) >= 1e6:
        x = [x / 1e6 for x in x]
        x_label = '[Mio]'
    if max(*y1, *[x for x in y1_lim or [] if x]) >= 1e9:
        y1 = [x / 1e9 for x in y1]
        if y1_lim:
            y1_lim = [(x / 1e9) if x else None for x in y1_lim]
        y1_label = '[Bn]'
    elif max(*y1, *[x for x in y1_lim or [] if x]) >= 1e6:
        y1 = [x / 1e6 for x in y1]
        if y1_lim:
            y1_lim = [(x / 1e6) if x else None for x in y1_lim]
        y1_label = '[Mio]'
    if y2 is not None and max(*y2, *[x for x in y2_lim or [] if x]) >= 1e9:
        y2 = [x / 1e9 for x in y2]
        if y2_lim:
            y2_lim = [(x / 1e9) if x else None for x in y2_lim]
        y2_label = '[Bn]'
    elif y2 is not None and max(*y2, *[x for x in y2_lim or [] if x]) >= 1e6:
        y2 = [x / 1e6 for x in y2]
        if y2_lim:
            y2_lim = [(x / 1e6) if x else None for x in y2_lim]
        y2_label = '[Mio]'
    return x, x_label, y1, y1_label, y2, y2_label, y1_lim, y2_lim


def export_legend(items, filepath="legend", format="png", expand=[-4, -4, 4, 4]):
    labels, colors = zip(*items)
    labels = [TEXT_MAPPING.get(x, x) for x in labels]
    handles = [plt.Line2D([], [], linewidth=3, color=colors[i]) for i in range(len(colors))]
    legend = plt.legend(handles, labels, loc=3, framealpha=0, frameon=False, ncol=1)
    plt.axis('off')
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    # timestamp = time.strftime('%m%d-%H%M%S')
    path = f'{filepath}.{format}'
    fig.savefig(path, dpi="figure", bbox_inches=bbox)


def save_plot(fig, filename, plots_dir=DEFAULT_PLOTS_DIR, format="png"):
    timestamp = time.strftime('%m%d-%H%M%S')
    fig.tight_layout()
    path = os.path.join(plots_dir, f'{timestamp}-{filename}')
    fig.savefig(f'{path}.{format}')
    return path
