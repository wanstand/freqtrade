# QTPyLib: Quantitative Trading Python Library
# https://github.com/ranaroussi/qtpylib
#
# Copyright 2016-2018 Ran Aroussi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import warnings
from datetime import datetime, timedelta
import math

import numpy as np  # noqa
import numba as nb
import pandas as pd  # noqa
import talib.abstract as ta
from scipy.interpolate import CubicSpline
from pandas import DataFrame, Series
from freqtrade.optimize.optimize_reports import store_backtest_series_data, load_backtest_series_data

# =============================================
warnings.simplefilter(action="ignore", category=RuntimeWarning)


# =============================================


@nb.jit(nopython=True)
def _mark_mm_core(
        baseline: np.ndarray,
        baseline_refer: np.ndarray,
        value: np.ndarray,
        marks: [],
        dates: np.ndarray,
        max_col: np.ndarray,
        min_col: np.ndarray,
        max_mark_cols: [],
        min_mark_cols: [],
        base_mark_cols: [],
        circle_col: np.ndarray,
        prev_max_col: np.ndarray,
        prev_min_col: np.ndarray,
        prev_max_mark_cols: [],
        prev_min_mark_cols: [],
        prev_base_mark_cols: [],
        prev_prev_max_col: np.ndarray,
        prev_prev_min_col: np.ndarray,
        side_cnt: int
):
    n = len(baseline)
    mark_num = len(marks)

    # 检测周期边界
    cycle_boundaries = [0]
    for i in range(1, n):
        if baseline_refer[i] < baseline[i] and baseline_refer[i - 1] >= baseline[i - 1]:  # 下穿检测
            cycle_boundaries.append(i)
        if side_cnt > 1:
            if baseline_refer[i] > baseline[i] and baseline_refer[i - 1] <= baseline[i - 1]:  # 上穿检测
                cycle_boundaries.append(i)

    cycle_boundaries.append(n)
    num_cycles = len(cycle_boundaries) - 1

    # 存储历史极值 (最多保留前两个周期)
    prev_max_vals = np.full(2, np.nan)
    prev_min_vals = np.full(2, np.nan)
    prev_max_marks = np.full(mark_num, np.nan)
    prev_min_marks = np.full(mark_num, np.nan)
    prev_base_marks = np.full(mark_num, np.nan)

    for cycle_idx in range(num_cycles):
        start = cycle_boundaries[cycle_idx]
        end = cycle_boundaries[cycle_idx + 1]

        # 计算当前周期极值
        current_max = value[start]
        current_min = value[start]
        current_max_marks = np.full(mark_num, np.nan)
        current_min_marks = np.full(mark_num, np.nan)
        current_base_marks = np.full(mark_num, np.nan)
        for v in range(mark_num):
            current_max_marks[v] = marks[v][start]
            current_min_marks[v] = marks[v][start]
            current_base_marks[v] = marks[v][start]
        for j in range(start, end):
            for v in range(mark_num):
                current_max_marks[v] = max(current_max_marks[v], marks[v][j])
                current_min_marks[v] = min(current_min_marks[v], marks[v][j])
            if value[j] > current_max:
                current_max = value[j]
            #    current_max_mark = max(current_max_mark, mark[j])
            #    current_min_mark = min(current_min_mark, mark[j])
            if value[j] < current_min:
                current_min = value[j]
            #    current_base_mark = mark[j]
            max_col[j] = current_max
            min_col[j] = current_min
            for v in range(mark_num):
                max_mark_cols[v][j] = current_max_marks[v]
                min_mark_cols[v][j] = current_min_marks[v]
                base_mark_cols[v][j] = current_base_marks[v]

        # 记录周期起始日期
        circle_col[start:end] = dates[start]

        # 填充前一周期的极值
        if cycle_idx >= 1:
            prev_max = prev_max_vals[0]
            prev_min = prev_min_vals[0]
            prev_max_col[start:end] = prev_max
            prev_min_col[start:end] = prev_min
            for v in range(mark_num):
                prev_max_mark_cols[v][start:end] = prev_max_marks[v]
                prev_min_mark_cols[v][start:end] = prev_min_marks[v]
                prev_base_mark_cols[v][start:end] = prev_base_marks[v]

        # 填充前前周期的极值
        if cycle_idx >= 2:
            prev_prev_max = prev_max_vals[1]
            prev_prev_min = prev_min_vals[1]
            prev_prev_max_col[start:end] = prev_prev_max
            prev_prev_min_col[start:end] = prev_prev_min

        # 更新历史极值队列
        if cycle_idx >= 1:
            prev_max_vals[1] = prev_max_vals[0]
            prev_min_vals[1] = prev_min_vals[0]
        prev_max_vals[0] = current_max
        prev_min_vals[0] = current_min
        for v in range(mark_num):
            prev_max_marks[v] = current_max_marks[v]
            prev_min_marks[v] = current_min_marks[v]
            prev_base_marks[v] = current_base_marks[v]


@nb.jit(nopython=False)
def numba_mark_super_volume(volume, volume_mean, diff, threshold=5.0, step=1.0):
    n = len(volume)
    result = np.zeros(n)
    for i in range(n):
        d = diff[i]
        if i > 0:
            if result[i - 1] > 0:
                d = abs(d)
                result[i] = result[i - 1] - step
                if result[i] < 0:
                    result[i] = 0
            elif result[i - 1] < 0:
                d = (0 - abs(d))
                result[i] = result[i - 1] + step
                if result[i] > 0:
                    result[i] = 0
        if volume[i] > (volume_mean[i] * threshold):
            r = (volume[i] / volume_mean[i])
            if d > 0:
                if i > 0:
                    result[i] = result[i - 1] + r
                else:
                    result[i] = r
            else:
                if i > 0:
                    result[i] = result[i - 1] - r
                else:
                    result[i] = 0 - r

    return result


@nb.jit(nopython=True)
def numba_cumulative_quantities(prices, quantities, target_ratio=0.1, timeperiod=144):
    vector = np.ones(timeperiod)
    for i in range(timeperiod - 1):
        vector[timeperiod - 2 - i] = (vector[timeperiod - 1 - i]) * (timeperiod - 1) / (timeperiod + 1)

    # print(vector)
    n = len(prices)
    result = np.empty(n)
    for i in range(n):
        start = max(0, i - timeperiod)
        size = i + 1 - start
        window_prices = prices[start:i + 1]
        window_quantities = quantities[start:i + 1] * vector[timeperiod - size:] * 2
        total = window_quantities.sum()
        target = total * target_ratio
        if target <= 0:
            result[i] = np.nan
            continue
        # 按价格排序并累积
        sorted_indices = np.argsort(window_prices)
        cum_quantities = np.cumsum(window_quantities[sorted_indices])
        mask = cum_quantities >= target
        if np.any(mask):
            first_valid = np.where(mask)[0][0]
            result[i] = window_prices[sorted_indices[first_valid]]
        else:
            result[i] = np.nan
    return result


# @nb.jit(nopython=False)
def _mark_flines_core(
        volume_ratio: np.ndarray,
        dates: np.ndarray,
        flines: np.ndarray,
        fline_cols: np.ndarray,
        fline_date_cols: np.ndarray,
        col_num,
        threshold=5.0
):
    n = len(volume_ratio)

    # 检测周期边界
    boundaries = [0]
    for i in range(0, n):
        if volume_ratio[i] > threshold:
            boundaries.append(i)
    boundaries.append(n)
    nums = len(boundaries) - 1

    for idx in range(nums):
        if idx > 0:
            last_idxs = boundaries[idx - 1]
        else:
            last_idxs = boundaries[idx]
        idxs = boundaries[idx]
        idxe = boundaries[idx + 1]
        if idx == 0:
            continue
        fline = flines[idxs]
        dt = dates[idxs]
        size = idxe - idxs

        last_flines = []
        last_dates = []
        for cidx in range(0, col_num):
            l = fline_cols[cidx, last_idxs]
            if l > 0:
                last_flines.append(l)
                last_dates.append(fline_date_cols[cidx, last_idxs])
        llen = len(last_flines)
        if size > 3:  # 1和2都是在变化中
            if llen == 0:
                last_flines.append(fline)
                last_dates.append(dt)
                llen = llen + 1
            else:
                lplace = llen
                for lidx in range(0, llen):
                    if fline < last_flines[lidx]:
                        lplace = lidx
                        break

                # print('last_flines:', last_flines)
                # print('last_dates:', last_dates)
                # print("llen=" + str(llen) + ",lplace=" + str(lplace) + ",fline=", fline)
                # 首先看是否合并
                ratio_1 = 100.0000
                ratio_2 = 100.0000
                if lplace > 0:
                    ratio_1 = (fline / last_flines[lplace - 1])
                    # print('ratio_1:', (fline / last_flines[lplace - 1]))
                if lplace < llen:
                    ratio_2 = (last_flines[lplace] / fline)
                    # print('ratio_2:', (fline / last_flines[lplace - 1]))
                if min(ratio_1, ratio_2) < 1.02:
                    if ratio_1 < ratio_2:
                        last_flines[lplace - 1] = fline
                        last_dates[lplace - 1] = dt
                    else:
                        last_flines[lplace] = fline
                        last_dates[lplace] = dt
                elif llen < col_num:  # 行不够直接插入
                    # print('insert-line')
                    last_flines.append(last_flines[llen - 1])
                    last_dates.append(last_dates[llen - 1])
                    for i in range(lplace, llen):
                        last_flines[llen + lplace - i] = last_flines[llen + lplace - i - 1]
                        last_dates[llen + lplace - i] = last_dates[llen + lplace - i - 1]
                    llen = llen + 1
                    last_flines[lplace] = fline
                    last_dates[lplace] = dt
                else:
                    # print('adjust-line')
                    # 删除小的
                    old_dt = dt
                    min_gap = 1.00
                    old_place = 0
                    for i in range(1, llen):
                        gap = (last_flines[i] / last_flines[i - 1]) - 1
                        if gap < min_gap:
                            min_gap = gap
                            old_place = i
                    if min_gap > 0.015:
                        for i in range(0, llen):
                            if last_dates[i] < old_dt:
                                old_place = i
                                old_dt = last_dates[i]
                    else:
                        if old_place >= (llen - 2):
                            old_place = llen - 2
                        elif old_place <= 1:
                            old_place = 1
                        else:
                            if (last_flines[old_place] / last_flines[old_place - 1]) \
                                    > (last_flines[old_place + 2] / last_flines[old_place + 1]):
                                old_place = old_place + 1
                    if lplace > old_place:
                        for i in range(old_place, lplace - 1):
                            last_flines[i] = last_flines[i + 1]
                            last_dates[i] = last_dates[i + 1]
                        last_flines[lplace - 1] = fline
                        last_dates[lplace - 1] = dt
                    elif lplace < old_place:
                        for i in range(lplace, old_place):
                            last_flines[old_place + lplace - i] = last_flines[old_place + lplace - i - 1]
                            last_dates[old_place + lplace - i] = last_dates[old_place + lplace - i - 1]
                        last_flines[lplace] = fline
                        last_dates[lplace] = dt
                    else:  # 正好等于
                        last_flines[lplace] = fline
                        last_dates[lplace] = dt
        for col_idx in range(0, llen):
            fline_cols[col_idx, idxs:idxe] = last_flines[col_idx]
            fline_date_cols[col_idx, idxs:idxe] = last_dates[col_idx]
        for col_idx in range(llen, col_num):
            fline_cols[col_idx, idxs:idxe] = np.nan
            fline_date_cols[col_idx, idxs:idxe] = 0


@nb.jit(nopython=True)
def _mark_boundary_core(
        volume_ratio: np.ndarray,
        open: np.ndarray,
        close: np.ndarray,
        dates: np.ndarray,
        bound_date: np.ndarray,
        fline_col: np.ndarray,
        prev_fline_col: np.ndarray,
        prev_prev_fline_col: np.ndarray,
        direction: np.ndarray,
        prev_direction: np.ndarray,
        threshold=5.0
):
    n = len(volume_ratio)

    # 检测周期边界
    boundaries = [0]
    for i in range(0, n):
        if volume_ratio[i] > threshold:
            boundaries.append(i)
    boundaries.append(n)
    nums = len(boundaries) - 1

    for idx in range(nums):
        idxs = boundaries[idx]
        idxe = boundaries[idx + 1]
        # 第一个是没有值的
        if idxs < 6:
            fline = np.nan
            prev_fline = np.nan
            prev_prev_fline = np.nan
            direct = 0
            prev_direct = np.nan
            bound_d = dates[idxs]
        else:
            fline = fline_col[idxs - 1]
            prev_fline = prev_fline_col[idxs - 1]
            prev_prev_fline = prev_prev_fline_col[idxs - 1]
            prev_direct = prev_direction[idxs - 1]
            this_max = max(open[idxs], close[idxs])
            this_min = min(open[idxs], close[idxs])
            last_size = boundaries[idx] - boundaries[idx - 1]

            this_fline = this_min
            if open[idxs] > open[idxs - 7]:
                direct = 1
                this_fline = this_max
                if open[idxs] > close[idxs]:  # 回找2个candle
                    if open[idxs - 1] > close[idxs - 1] and volume_ratio[idxs - 1] > 4:
                        this_fline = max(this_fline, open[idxs - 1])
                        if open[idxs - 2] > close[idxs - 2] and volume_ratio[idxs - 2] > 4:
                            this_fline = max(this_fline, open[idxs - 2])
            else:
                direct = -1
                if open[idxs] < close[idxs]:  # 回找2个candle
                    if open[idxs - 1] < close[idxs - 1] and volume_ratio[idxs - 1] > 4:
                        this_fline = min(this_fline, open[idxs - 1])
                        if open[idxs - 2] > close[idxs - 2] and volume_ratio[idxs - 2] > 4:
                            this_fline = min(this_fline, open[idxs - 2])
            bound_d = dates[idxs]
            if np.isnan(fline):  # 第一次填充
                fline = this_fline
            else:
                if last_size < 3:  # 连续大于门限值
                    bound_d = bound_date[idxs - 1]
                    direct = direction[idxs - 1]
                    if direct > 0:
                        fline = max(this_max, fline)
                    else:
                        fline = min(this_min, fline)
                else:
                    if abs(this_fline - fline) > (fline / 100):
                        prev_prev_fline = prev_fline
                        prev_fline = fline
                        fline = this_fline
                    else:  # 改变太小忽略
                        bound_d = bound_date[idxs - 1]
                    prev_direct = direction[idxs - 1]  # 无论如何都改变方向

        fline_col[idxs:idxe] = fline
        prev_fline_col[idxs:idxe] = prev_fline
        prev_prev_fline_col[idxs:idxe] = prev_prev_fline
        direction[idxs:idxe] = direct
        prev_direction[idxs:idxe] = prev_direct
        bound_date[idxs:idxe] = bound_d


@nb.jit(nopython=True)
def _numba_cumulative_quantities_ohlc(
        price_open,
        price_close,
        quantities,
        target_ratio=0.1,
        timeperiod=144
) -> np.ndarray:
    """
    Calculates the price at which a target cumulative quantity is reached,
    assuming quantities are linearly distributed between open and close prices.

    Args:
        price_open (np.ndarray): Array of opening prices.
        price_close (np.ndarray): Array of closing prices.
        quantities (np.ndarray): Array of quantities.
        target_ratio (float): The target ratio of total quantity.
        timeperiod (int): The lookback window period.

    Returns:
        np.ndarray: An array of calculated prices. NaN if the target
                    is not met.
    """
    # The weighting vector calculation remains the same.
    # This creates a geometrically decaying weight.
    vector = np.ones(timeperiod)
    if timeperiod > 1:
        for i in range(timeperiod - 1):
            vector[timeperiod - 2 - i] = (vector[timeperiod - 1 - i]) * (timeperiod - 1) / (timeperiod + 1)

    n = len(price_open)
    result = np.empty(n)

    # Pre-allocate arrays for events to avoid allocation inside the loop,
    # which is faster in Numba. Maximum possible events for a window is 2 * timeperiod.
    max_events = 2 * timeperiod
    event_prices = np.empty(max_events)
    # Event types: 0=start of a range, 1=end of a range, 2=point quantity
    event_types = np.empty(max_events, dtype=np.uint8)
    event_values = np.empty(max_events)

    for i in range(n):
        # Use a standard sliding window implementation.
        # The window size is `min(i + 1, timeperiod)`.
        start = max(0, i - timeperiod + 1)
        size = i - start + 1

        window_open = price_open[start:i + 1]
        window_close = price_close[start:i + 1]

        # Apply the weighting vector to the quantities in the window.
        window_quantities = quantities[start:i + 1] * vector[timeperiod - size:] * 2

        total_qty = window_quantities.sum()
        target = total_qty * target_ratio

        if total_qty <= 0:
            result[i] = np.nan
            continue

        # --- Create Events from Bars in the Window ---
        num_events = 0
        for k in range(size):
            p_open = window_open[k]
            p_close = window_close[k]
            qty = window_quantities[k]

            if qty <= 0:  # Ignore bars with no quantity
                continue

            if p_open == p_close:
                # This is a "point" event, where all quantity is at a single price.
                event_prices[num_events] = p_open
                event_types[num_events] = 2  # type: point
                event_values[num_events] = qty
                num_events += 1
            else:
                # This is a "range" event, quantity is distributed linearly.
                p_low = min(p_open, p_close)
                p_high = max(p_open, p_close)
                density = qty / (p_high - p_low)

                # Create a 'start' event at the low price
                event_prices[num_events] = p_low
                event_types[num_events] = 0  # type: start
                event_values[num_events] = density
                num_events += 1

                # Create an 'end' event at the high price
                event_prices[num_events] = p_high
                event_types[num_events] = 1  # type: end
                event_values[num_events] = density
                num_events += 1

        if num_events == 0:
            result[i] = np.nan
            continue

        # --- Sort events by price to process them in order ---
        # We only sort the active part of the arrays.
        active_event_prices = event_prices[:num_events]
        sorted_indices = np.argsort(active_event_prices)

        # --- Process Events to Find the Target Price ---
        cum_qty = 0.0
        current_density = 0.0
        # Get the price of the very first event to initialize
        first_event_idx = sorted_indices[0]
        last_price = event_prices[first_event_idx]

        found = False
        for j in range(num_events):
            event_idx = sorted_indices[j]
            price = event_prices[event_idx]
            etype = event_types[event_idx]
            value = event_values[event_idx]

            # Calculate quantity accumulated in the price interval since the last event
            price_delta = price - last_price
            if current_density > 0 and price_delta > 0:
                qty_in_interval = current_density * price_delta

                # Check if the target is met within this continuous interval
                if cum_qty < target and cum_qty + qty_in_interval >= target:
                    price_needed = (target - cum_qty) / current_density
                    result[i] = last_price + price_needed
                    found = True
                    break

                cum_qty += qty_in_interval

            # Process the event itself at the current price point.
            # Unstable sort order for same-priced events is fine because price_delta is 0.
            if etype == 2:  # Point event
                # Check if the target is met by this point quantity
                if cum_qty < target and cum_qty + value >= target:
                    result[i] = price
                    found = True
                    break
                cum_qty += value
            elif etype == 0:  # Start event
                current_density += value
            elif etype == 1:  # End event
                current_density -= value

            last_price = price

        if not found:
            # If the loop finishes and the target hasn't been found,
            # it means the target is greater than the total available quantity.
            result[i] = np.nan

    return result


# Numba 优化的核心计算部分
@nb.jit(nopython=True)
def _calculate_rs_values_numba(log_returns):
    """
    Core R/S calculation logic, optimized with Numba.
    This function computes the R/S values for different segment lengths.
    """
    n_values = []
    rs_values = []

    max_k = len(log_returns) // 2

    for k in range(2, max_k + 1):
        num_segments = len(log_returns) // k

        if num_segments == 0:
            continue

        rs_per_segment_list = []

        for i in range(num_segments):
            start_idx = i * k
            end_idx = start_idx + k
            segment = log_returns[start_idx:end_idx]

            if len(segment) < 2:
                continue

            mean_segment = np.mean(segment)
            cumulative_deviation = np.cumsum(segment - mean_segment)

            R = np.max(cumulative_deviation) - np.min(cumulative_deviation)
            S = np.std(segment)

            if S == 0:
                continue

            rs_per_segment_list.append(R / S)

        if rs_per_segment_list:
            avg_rs = np.mean(np.array(rs_per_segment_list))
            n_values.append(k)
            rs_values.append(avg_rs)

    # 返回 NumPy 数组，以便在外部进行 polyfit
    return np.array(n_values), np.array(rs_values)


"""
def mark_trends(self, df: DataFrame, value: str):
    df[value + '-diff'] = df[value].diff()
    df[value + '-direction'] = np.where(df[value + '-diff'] > 0, 1, np.where(df[value + '-diff'] < 0, -1, 0))
    # 处理连续上涨
    up = (df[value + '-direction'] == 1)
    up_group = (up != up.shift()).cumsum()  # 生成分组标识
    df[value + '-trend-count'] = df.groupby(up_group).cumcount() + 1  # 组内累计计数
    df.drop([value + '-diff', value + '-direction'], axis=1, inplace=True)
"""


def numba_cumulative_quantities_ohlc(
        config: dict,
        price_open: pd.Series,
        price_close: pd.Series,
        quantities: pd.Series,
        target_ratio=0.1,
        timeperiod=144
) -> np.ndarray:
    ver = f"_ratio_{target_ratio}_timeperiod_{timeperiod}_0001"
    src = f"open_{price_open.name}_close_{price_close.name}_quantities_{quantities.name}"
    data = _auto_load(config, "numba_cumulative_quantities_ohlc", src, ver)
    if data is None:
        data = _numba_cumulative_quantities_ohlc(price_open.values, price_close.values, quantities.values,
                                                 target_ratio, timeperiod)
        _auto_save(config, data, "numba_cumulative_quantities_ohlc", src, ver)
    return data


def mark_flines_numba(config: dict, df: DataFrame, threshold=5.0, col_num=9):
    ver = f"threshold_{threshold}_col_num_{col_num}_0001"
    n = len(df['date'])
    fline_cols = np.empty((col_num, n), dtype=np.float64)
    fline_date_cols = np.empty((col_num, n), dtype=np.float64)
    fullloaded = True
    print("mark_flines_numba")
    for i in range(0, col_num):
        fline_col = _auto_load(config, "mark_flines_numba", f"fline_col_{i}", ver)
        fline_date_col = _auto_load(config, "mark_flines_numba", f"fline_date_col_{i}", ver)
        fullloaded = fullloaded and (fline_col is not None) and (fline_date_col is not None)
        if not fullloaded:
            break
        fline_cols[i] = fline_col
        fline_date_cols[i] = fline_date_col

    if not fullloaded:
        # 将关键列转换为 NumPy 数组
        volume_ratio = df['volume-ratio'].values
        flines = df['bound-line'].values
        dates = df['date'].view('int64').values

        for i in range(0, col_num):
            fline_cols[i] = np.zeros(n)
            fline_date_cols[i] = np.zeros(n)

        _mark_flines_core(
            volume_ratio,
            dates,
            flines,
            fline_cols,
            fline_date_cols,
            col_num=col_num,
            threshold=threshold
        )

    if not fullloaded:
        for i in range(0, col_num):
            _auto_save(config, fline_cols[i], "mark_flines_numba", f"fline_col_{i}", ver)
            _auto_save(config, fline_date_cols[i], "mark_flines_numba", f"fline_date_col_{i}", ver)

    for i in range(0, col_num):
        df[f'fline-{i + 1}'] = fline_cols[i]
        df[f'fline-date-{i + 1}'] = fline_date_cols[i]

    for i in range(0, col_num - 1):
        df[f'dline-{i + 1}'] = (df[f'fline-{i + 2}'] - df[f'fline-{i + 1}'])
    df[f'dline-0'] = (df[f'fline-2'] - df[f'fline-1'])
    df[f'dline-{col_num}'] = (df[f'fline-{col_num}'] - df[f'fline-{col_num - 1}'])


def mark_boundary_numba(config: dict, df: DataFrame, threshold=5.0):
    ver = f"threshold_{threshold}_0001"
    bound_date = _auto_load(config, "mark_boundary_numba", "bound_date", ver)
    fline_col = _auto_load(config, "mark_boundary_numba", "fline_col", ver)
    prev_fline_col = _auto_load(config, "mark_boundary_numba", "prev_fline_col", ver)
    prev_prev_fline_col = _auto_load(config, "mark_boundary_numba", "prev_prev_fline_col", ver)
    direction = _auto_load(config, "mark_boundary_numba", "direction", ver)
    prev_direction = _auto_load(config, "mark_boundary_numba", "prev_direction", ver)
    fullloaded = (bound_date is not None) \
                 and (fline_col is not None) \
                 and (prev_fline_col is not None) \
                 and (prev_prev_fline_col is not None) \
                 and (direction is not None) \
                 and (prev_direction is not None)

    if not fullloaded:
        n = len(df['open'])
        # 将关键列转换为 NumPy 数组
        volume_ratio = df['volume-ratio'].values
        open = df['open'].values
        close = df['close'].values
        dates = df['date'].view('int64').values

        # 预分配结果数组 (假设已经预先创建了这些列)
        bound_date = np.zeros(n)
        fline_col = np.zeros(n)
        prev_fline_col = np.zeros(n)
        prev_prev_fline_col = np.zeros(n)
        direction = np.zeros(n)
        prev_direction = np.zeros(n)

        # 调用 Numba 优化核心逻辑
        _mark_boundary_core(
            volume_ratio,
            open,
            close,
            dates,
            bound_date,
            fline_col,
            prev_fline_col,
            prev_prev_fline_col,
            direction,
            prev_direction,
            threshold=threshold
        )

    # 将结果写回 DataFrame
    df['bound-date'] = bound_date
    df['bound-line'] = fline_col
    df['bound-prev-line'] = prev_fline_col
    df['bound-prev-prev-line'] = prev_prev_fline_col
    df['bound-direction'] = direction
    df['bound-prev-direction'] = prev_direction

    if not fullloaded:
        _auto_save(config, bound_date, "mark_boundary_numba", "bound_date", ver)
        _auto_save(config, fline_col, "mark_boundary_numba", "fline_col", ver)
        _auto_save(config, prev_fline_col, "mark_boundary_numba", "prev_fline_col", ver)
        _auto_save(config, prev_prev_fline_col, "mark_boundary_numba", "prev_prev_fline_col", ver)
        _auto_save(config, direction, "mark_boundary_numba", "direction", ver)
        _auto_save(config, prev_direction, "mark_boundary_numba", "prev_direction", ver)


def mark_mm_numba(config: dict, df: DataFrame, baseline: str, baselinevar: str, value: str, marks: [],
                  baseline_refer='', side_cnt=1):
    ver = f"baseline_{baseline}_var_{baselinevar}_{baseline_refer}_side_{side_cnt}_0001"
    max_col = _auto_load(config, "mark_mm_numba", value + "-max", ver)
    min_col = _auto_load(config, "mark_mm_numba", value + "-min", ver)
    circle_col = _auto_load(config, "mark_mm_numba", value + "-circle", ver)
    prev_max_col = _auto_load(config, "mark_mm_numba", value + "-prev-max", ver)
    prev_min_col = _auto_load(config, "mark_mm_numba", value + "-prev-min", ver)
    prev_prev_max_col = _auto_load(config, "mark_mm_numba", value + "-prev-prev-max", ver)
    prev_prev_min_col = _auto_load(config, "mark_mm_numba", value + "-prev-prev-min", ver)
    fullloaded = (max_col is not None) \
                 and (min_col is not None) \
                 and (circle_col is not None) \
                 and (prev_max_col is not None) \
                 and (prev_min_col is not None) \
                 and (prev_prev_max_col is not None) \
                 and (prev_prev_min_col is not None)

    marks_size = len(marks)
    max_mark_cols = [None] * marks_size
    min_mark_cols = [None] * marks_size
    base_mark_cols = [None] * marks_size
    prev_max_mark_cols = [None] * marks_size
    prev_min_mark_cols = [None] * marks_size
    prev_base_mark_cols = [None] * marks_size
    if fullloaded:
        for i in range(len(marks)):
            mark = marks[i]
            max_mark_cols[i] = _auto_load(config, "mark_mm_numba", value + '-' + mark + "-max", ver)
            min_mark_cols[i] = _auto_load(config, "mark_mm_numba", value + '-' + mark + "-min", ver)
            base_mark_cols[i] = _auto_load(config, "mark_mm_numba", value + '-' + mark + "-base", ver)
            prev_max_mark_cols[i] = _auto_load(config, "mark_mm_numba", value + '-' + mark + "-prev-max", ver)
            prev_min_mark_cols[i] = _auto_load(config, "mark_mm_numba", value + '-' + mark + "-prev-min", ver)
            prev_base_mark_cols[i] = _auto_load(config, "mark_mm_numba", value + '-' + mark + "-prev-base", ver)
            fullloaded = fullloaded \
                         and (max_mark_cols[i] is not None) \
                         and (min_mark_cols[i] is not None) \
                         and (base_mark_cols[i] is not None) \
                         and (prev_max_mark_cols[i] is not None) \
                         and (prev_min_mark_cols[i] is not None) \
                         and (prev_base_mark_cols[i] is not None)
            if not fullloaded:
                break

    if not fullloaded:
        n = len(df[value])
        # 将关键列转换为 NumPy 数组
        if baseline == "ZERO":
            baseline_arr = np.zeros(n)
        else:
            baseline_arr = df[baseline].values
        value_arr = df[value].values
        if baseline_refer == '':
            baseline_refer_arr = value_arr
        else:
            baseline_refer_arr = df[baseline_refer].values
        mark_arrs = []
        for i in range(len(marks)):
            mark_arrs.append(df[marks[i]].values)
        dates = df['date'].view('int64').values

        # 预分配结果数组 (假设已经预先创建了这些列)
        max_col = np.zeros(n)
        min_col = np.zeros(n)
        for i in range(len(marks)):
            max_mark_cols[i] = (np.zeros(n))
            min_mark_cols[i] = (np.zeros(n))
            base_mark_cols[i] = (np.zeros(n))
            prev_max_mark_cols[i] = (np.zeros(n))
            prev_min_mark_cols[i] = (np.zeros(n))
            prev_base_mark_cols[i] = (np.zeros(n))
        circle_col = np.zeros(n, dtype=np.int64)
        prev_max_col = np.zeros(n)
        prev_min_col = np.zeros(n)
        prev_prev_max_col = np.zeros(n)
        prev_prev_min_col = np.zeros(n)

        # 调用 Numb
        # a 优化核心逻辑
        _mark_mm_core(
            baseline_arr, baseline_refer_arr, value_arr, mark_arrs, dates,
            max_col, min_col, max_mark_cols, min_mark_cols, base_mark_cols,
            circle_col,
            prev_max_col, prev_min_col, prev_max_mark_cols, prev_min_mark_cols, prev_base_mark_cols,
            prev_prev_max_col, prev_prev_min_col,
            side_cnt
        )
    if not fullloaded:
        _auto_save(config, max_col, "mark_mm_numba", value + "-max", ver)
        _auto_save(config, min_col, "mark_mm_numba", value + "-min", ver)
        _auto_save(config, circle_col, "mark_mm_numba", value + "-circle", ver)
        _auto_save(config, prev_max_col, "mark_mm_numba", value + "-prev-max", ver)
        _auto_save(config, prev_min_col, "mark_mm_numba", value + "-prev-min", ver)
        _auto_save(config, prev_prev_max_col, "mark_mm_numba", value + "-prev-prev-max", ver)
        _auto_save(config, prev_prev_min_col, "mark_mm_numba", value + "-prev-prev-min", ver)

    # 将结果写回 DataFrame
    df[value + '-max'] = max_col
    df[value + '-min'] = min_col
    df[baseline + '-circle'] = circle_col
    df[value + '-prev-max'] = prev_max_col
    df[value + '-prev-min'] = prev_min_col
    df[value + '-prev-prev-max'] = prev_prev_max_col
    df[value + '-prev-prev-min'] = prev_prev_min_col

    for i in range(len(marks)):
        mark = marks[i]
        df[value + '-' + mark + '-max'] = max_mark_cols[i]
        df[value + '-' + mark + '-min'] = min_mark_cols[i]
        df[value + '-' + mark + '-base'] = base_mark_cols[i]
        df[value + '-' + mark + '-prev-max'] = prev_max_mark_cols[i]
        df[value + '-' + mark + '-prev-min'] = prev_min_mark_cols[i]
        df[value + '-' + mark + '-prev-base'] = prev_base_mark_cols[i]
        if not fullloaded:
            _auto_save(config, max_mark_cols[i], "mark_mm_numba", value + '-' + mark + "-max", ver)
            _auto_save(config, min_mark_cols[i], "mark_mm_numba", value + '-' + mark + "-min", ver)
            _auto_save(config, base_mark_cols[i], "mark_mm_numba", value + '-' + mark + "-base", ver)
            _auto_save(config, prev_max_mark_cols[i], "mark_mm_numba", value + '-' + mark + "-prev-max", ver)
            _auto_save(config, prev_min_mark_cols[i], "mark_mm_numba", value + '-' + mark + "-prev-min", ver)
            _auto_save(config, prev_base_mark_cols[i], "mark_mm_numba", value + '-' + mark + "-prev-base", ver)


def calc_trend_ratio(config: dict, s: Series, wavelenth: int, multi: int) -> np.ndarray:
    ver = f"wavelenth_{wavelenth}_multi_{multi}_v_0001"
    # ver = f"wavelenth_{wavelenth}_multi_{multi}_v_0002sqrt"
    load = _auto_load(config, "calc_trend_ratio", s.name, ver)
    if load is not None:
        return load
    src = s.values
    n = len(src)
    dest = np.zeros(n)
    for i in range(wavelenth * (multi + 1), n):
        # 切片提取从i开始的k+1个元素
        sub_src = src[np.arange(i - wavelenth * (multi + 1), i + 1, wavelenth)]  # i, i+j, i+2j, ..., i+k
        diffs = sub_src[:-1] - sub_src[1:]
        mask = sub_src[:-1] < sub_src[1:]
        # 计算满足前一个元素小于后一个元素的次数总和
        negative = np.sum(diffs[mask])
        positive = np.sum(diffs) - negative
        dest[i] = np.sqrt((src[i] - src[i - (wavelenth * multi)]) * (src[i] - src[i - (wavelenth * multi)]) * 100 \
                          / src[i] / (positive - negative)) * (1 if src[i] > src[i - (wavelenth * multi)] else -1)
    _auto_save(config, dest, "calc_trend_ratio", s.name, ver)
    return dest


def calc_trend_ratio_ema(config: dict, s: Series,
                         wavelenth: int, multi: int, timeperoid: int) -> np.ndarray:
    ver = f"wavelenth_{wavelenth}_multi_{multi}_timeperoid_{timeperoid}_v_0001"  # 0001：没开平方版本
    # ver = f"wavelenth_{wavelenth}_multi_{multi}_timeperoid_{timeperoid}_v_0002sqrt"
    load = _auto_load(config, "calc_trend_ratio_ema", s.name, ver)
    if load is not None:
        return load
    ratio = calc_trend_ratio(config, s, wavelenth, multi)
    ema = ta.EMA(ratio, timeperoid=timeperoid)
    _auto_save(config, ema, "calc_trend_ratio_ema", s.name, ver)
    return ema


def _auto_load(config: dict, func: str,
               attribute: str, version: str) -> np.ndarray:
    version = version.replace(".", "x")
    if 'isbacktesting' in config and config['isbacktesting']:
        load = load_backtest_series_data(config, f"{func}_{attribute}", version)
        if load is not None:
            print(f"Load succ:{func} {attribute} {version}")
            return load
        print(f"Load fail:{func} {attribute} {version}")
    return None


def _auto_save(config: dict, data: np.ndarray, func: str,
               attribute: str, version: str):
    version = version.replace(".", "x")
    if 'isbacktesting' in config and config['isbacktesting']:
        store_backtest_series_data(config, pd.Series(data), f"{func}_{attribute}", version)


def calc_hurst_rs(config: dict, s: pd.Series, timeperoid=144) -> np.ndarray:
    ver = f"0001_timeperoid_{timeperoid}"
    load = _auto_load(config, "calc_hurst_rs", s.name, ver)
    if load is not None:
        return load
    data = s.values
    hurst_rs = np.zeros(len(data))
    for i in range(len(data)):
        if i < timeperoid:
            hurst_rs[i] = np.NaN
        else:
            price_series_arr = np.asarray(data[i - timeperoid:i])

            log_returns = np.diff(np.log(price_series_arr))

            if len(log_returns) < 50:
                print("Input series is too short for reliable R/S analysis. Minimum 50 log returns recommended.")
                return np.nan

            # 调用 Numba 优化的函数来获取 n_values 和 rs_values
            n_values, rs_values = _calculate_rs_values_numba(log_returns)

            if len(n_values) < 5:
                print("Not enough data points to perform R/S analysis. Consider a longer price series.")
                return np.nan

            # 这一步在 Numba 外部执行
            log_n = np.log(n_values)
            log_rs = np.log(rs_values)

            poly = np.polyfit(log_n, log_rs, 1)
            hurst_exponent = poly[0]

            hurst_exponent = np.clip(hurst_exponent, 0.0, 1.0)
            fractal_dimension = 2 - hurst_exponent
            hurst_rs[i] = hurst_exponent
    _auto_save(config, hurst_rs, "calc_hurst_rs", s.name, ver)
    return hurst_rs


def rationalQuadratic(src: pd.Series, lookback: int, relativeWeight: float, startAtBar: int):
    """
    vectorized calculate for rational quadratic curve
    :param src:
    :param lookback:
    :param relativeWeight:
    :param startAtBar:
    :return:
    """
    currentWeight = [0.0] * len(src)
    cumulativeWeight = 0.0
    for i in range(startAtBar + 2):
        y = src.shift(i, fill_value=0.0)
        w = (1 + (i ** 2 / (lookback ** 2 * 2 * relativeWeight))) ** -relativeWeight
        currentWeight += y.values * w
        cumulativeWeight += w
    val = currentWeight / cumulativeWeight
    val[:startAtBar + 1] = 0.0

    return val


def gaussian(src: pd.Series, lookback: int, startAtBar: int):
    """
    vectorized calculate for gaussian curve
    :param src:
    :param lookback:
    :param startAtBar:
    :return:
    """
    currentWeight = [0.0] * len(src)
    cumulativeWeight = 0.0
    for i in range(startAtBar + 2):
        y = src.shift(i, fill_value=0.0)
        w = math.exp(-(i ** 2) / (2 * lookback ** 2))
        currentWeight += y.values * w
        cumulativeWeight += w
    val = currentWeight / cumulativeWeight
    val[:startAtBar + 1] = 0.0

    return val
