import warnings
from datetime import datetime, timedelta
import math

import numpy as np  # noqa
import numba as nb
import pandas as pd  # noqa
import talib.abstract as ta
from scipy.interpolate import CubicSpline
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from freqtrade.optimize.optimize_reports import store_backtest_series_data, load_backtest_series_data


def statistic_prints(df: DataFrame, metadata: dict, wsconfig: dict, attrs: [], algo: str, period: str):
    pair = metadata['pair']
    pair = pair.replace("/", "-").replace(":", "_")
    timerange = metadata['timerange']
    df['date'] = pd.to_datetime(df['date'], unit='ns', utc=True)
    df_indexed = df.set_index('date')
    df_indexed.replace(np.nan, 0, inplace=True)
    df_groups = df_indexed.groupby(pd.Grouper(freq=period))
    for attr in attrs:
        for period_idx, group in df_groups:
            name = period_idx.strftime('%Y-%m')
            nv = None
            if algo == 'mean':
                nv = group[attr].mean()
            elif algo == 'stddev':
                nv = group[attr].std()
            print(f'{attr}-{algo} {name}' + ':' + f'{nv}')
        nv = None
        if algo == 'mean':
            nv = df_indexed[attr].mean()
        elif algo == 'stddev':
            nv = df_indexed[attr].std()
        print(f'{attr}-{algo} Total' + ':' + f'{nv}')


def draw_hist(series: pd.Series, attr: str, name: str):
    plt.figure(figsize=(10, 6))
    sns.histplot(series, kde=True, bins=30)  # kde=True 会叠加一个核密度估计曲线
    plt.title(f'Hist: {name} of {attr}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'images/{name}-{attr}.hist.png', dpi=200, bbox_inches='tight')
    plt.close()


def draw_qq(series: pd.Series, attr: str, name: str):
    plt.figure(figsize=(10, 6))
    sm.qqplot(series, line='s', fit=True)  # line='s' 绘制标准线，fit=True 拟合最佳直线
    plt.title(f'QQ: {name} of {attr}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'images/{name}-{attr}.qq.png', dpi=200, bbox_inches='tight')
    plt.close()


def draw_logqq(series: pd.Series, attr: str, name: str):
    plt.figure(figsize=(10, 6))
    sm.qqplot(np.log(series), line='s', fit=True)  # line='s' 绘制标准线，fit=True 拟合最佳直线
    plt.title(f'LogQQ: {name} of {attr}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'images/{name}-{attr}.logqq.png', dpi=200, bbox_inches='tight')
    plt.close()


def statistic_draws(df: DataFrame, metadata: dict, wsconfig: dict, attrs: [], algo: str):
    pair = metadata['pair']
    pair = pair.replace("/", "-").replace(":", "_")
    timerange = metadata['timerange']
    for attr in attrs:
        df.replace(np.nan, 0, inplace=True)
        if algo == 'hist':
            draw_hist(df[attr], attr, f'{pair}-{timerange}')
        elif algo == 'qq':
            draw_qq(df[attr], attr, f'{pair}-{timerange}')
        elif algo == 'logqq':
            draw_logqq(df[attr], attr, f'{pair}-{timerange}')
