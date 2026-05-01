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
from numba import njit
from numba.typed import List
import pandas as pd  # noqa
import talib.abstract as ta
from scipy.interpolate import CubicSpline
from pandas import DataFrame, Series
from freqtrade.optimize.optimize_reports import store_backtest_series_data, load_backtest_series_data

# =============================================
warnings.simplefilter(action="ignore", category=RuntimeWarning)


# =============================================
@nb.jit(nopython=True)
def _ema_kernel(data, span):
    n = data.shape[0]
    out = np.zeros(n)
    alpha = 2.0 / (span + 1.0)

    if n > 0:
        out[0] = data[0]  # 初始化
        for i in range(1, n):
            out[i] = data[i] * alpha + out[i - 1] * (1.0 - alpha)
    return out

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

    print(f'mark_boundary_core:n={n},nums={nums}')
    for idx in range(nums):
        idxs = boundaries[idx]
        idxe = boundaries[idx + 1]
        # 增量计算支持
        # print(f'fline_col[{idxe-1}]={float(fline_col[idxe-1].item())}')
        if fline_col[idxe - 1] != 0 and idxe - idxs > 5:
            print(f'mark_boundary: jumped{idxs}~{idxe}')
            continue
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
def _calc_mean(value: np.ndarray, mean: np.ndarray, init_value: np.float64, timeperiod: int, to_calc_length: int):
    n = len(value)
    start = max(n - to_calc_length, 1)
    if to_calc_length >= n:
        mean[0] = init_value
        if np.isnan(mean[0]):
            mean[0] = 0.0
        for i in range(1, timeperiod):
            if np.isnan(value[i]):
                mean[i] = mean[i - 1]
                continue
            mean[i] = mean[i - 1] + ((value[i] - mean[i - 1]) / i)
        start = timeperiod
    for i in range(start, n):
        if np.isnan(value[i]):
            mean[i] = mean[i - 1]
            continue
        mean[i] = mean[i - 1] + ((value[i] - mean[i - 1]) / timeperiod)


"""
@nb.jit(nopython=True)
def _mark_hit_band(hh: np.ndarray, ll: np.ndarray, c: np.ndarray, hit: np.ndarray, hitcnt: np.ndarray,
                   step: int, step_max: int, to_calc_length: int):
    n = len(hh)
    start = max(n - to_calc_length, 1)
    if np.isnan(hit[0]):
        hit[0] = 0
    for i in range(start, n):
        if c[i] > hh[i]:
            hit[i] = max(hit[i - 1], 0) + step
            if hit[i] > step_max:
                hit[i] = step_max
            hitcnt[i] = max(hitcnt[i - 1], 0) + 1
        elif c[i] < ll[i]:
            hit[i] = min(hit[i - 1], 0) - step
            if hit[i] < (0 - step_max):
                hit[i] = (0 - step_max)
            hitcnt[i] = min(hitcnt[i - 1], 0) - 1
        else:
            if hit[i - 1] < 0:
                hit[i] = hit[i - 1] + 1
                hitcnt[i] = hitcnt[i - 1] - 1
            elif hit[i - 1] > 0:
                hit[i] = hit[i - 1] - 1
                hitcnt[i] = hitcnt[i - 1] + 1
            else:
                hitcnt[i] = 0
                hit[i] = 0
                

@nb.jit(nopython=True)
def _mark_hit_line(line: np.ndarray, c: np.ndarray, hit: np.ndarray, step_max: int,
                   to_calc_length: int):
    n = len(line)
    start = max(n - to_calc_length, 1)
    if np.isnan(hit[0]):
        hit[0] = 0
    for i in range(start, n):
        if c[i] > line[i] and c[i - 1] <= line[i - 1]:
            hit[i] = step_max
        elif c[i] < line[i] and c[i - 1] >= line[i - 1]:
            hit[i] = (0 - step_max)
        elif hit[i - 1] != 0:
            hit[i] = hit[i - 1] - np.sign(hit[i - 1])
        else:
            hit[i] = 0
"""


@nb.jit(nopython=True)
def _mark_hit_band(hh: np.ndarray, ll: np.ndarray, c: np.ndarray, hit: np.ndarray, hitcnt: np.ndarray,
                   step: int, step_max: int, to_calc_length: int):
    n = len(hh)
    start = max(n - to_calc_length, 1)
    m_d = np.exp(-1 / int(step_max / 2))
    if np.isnan(hit[0]):
        hit[0] = 0
    for i in range(start, n):
        if c[i] > hh[i]:
            hit[i] = max(hit[i - 1], 0) + step
            hitcnt[i] = max(hitcnt[i - 1], 0) + 1
        elif c[i] < ll[i]:
            hit[i] = min(hit[i - 1], 0) - step
            hitcnt[i] = min(hitcnt[i - 1], 0) - 1
        else:
            if hit[i - 1] < 0:
                hit[i] = (hit[i - 1] * m_d)
                if hit[i] > -1:
                    hit[i] = 0
                else:
                    hit[i] = hit[i] + 1
                hitcnt[i] = hitcnt[i - 1] - 1
            elif hit[i - 1] > 0:
                hit[i] = (hit[i - 1] * m_d)
                if hit[i] < 1:
                    hit[i] = 0
                else:
                    hit[i] = hit[i] - 1
                hitcnt[i] = hitcnt[i - 1] + 1
            else:
                hitcnt[i] = 0
                hit[i] = 0


@nb.jit(nopython=True)
def _mark_hit_line(line: np.ndarray, c: np.ndarray, hit: np.ndarray, step_max: int,
                   to_calc_length: int):
    n = len(line)
    start = max(n - to_calc_length, 1)
    m_d = np.exp(-1 / int(step_max / 2))
    if np.isnan(hit[0]):
        hit[0] = 0
    for i in range(start, n):
        if c[i] > line[i] and c[i - 1] <= line[i - 1]:
            hit[i] = np.abs(hit[i - 1]) + 1
        elif c[i] < line[i] and c[i - 1] >= line[i - 1]:
            hit[i] = -np.abs(hit[i - 1]) - 1
        elif hit[i - 1] != 0:
            hit[i] = hit[i - 1] * m_d
            if np.abs(hit[i]) < 0.05:
                hit[i] = 0
        else:
            hit[i] = 0


@nb.jit(nopython=True)
def _mark_circle_base(side: np.ndarray, idx: np.ndarray, cnt: np.ndarray,
                      to_calc_length: int):
    n = len(side)
    start = max(n - to_calc_length, 1)
    if np.isnan(side[0]):
        side[0] = 0
    if np.isnan(idx[0]) or idx[0] == 0:
        idx[0] = 10000
    if np.isnan(cnt[0]) or cnt[0] == 0:
        cnt[0] = 1
    for i in range(start, n):
        if np.isnan(side[i]) or side[i] == 0:
            side[i] = side[i - 1]
        if side[i] != side[i - 1]:
            idx[i] = idx[i - 1] + 1
            cnt[i] = np.sign(side[i])
        else:
            idx[i] = idx[i - 1]
            cnt[i] = cnt[i - 1] + np.sign(side[i])


@nb.jit(nopython=True)
def _mark_circle_vertexs(cnts: np.ndarray, attrs: [], signs: [], bases: [], avgs: [], maxx: [], max_cnt: [], minn: [],
                         min_cnt: [],
                         to_calc_length: int):
    n = len(cnts)
    start = max(n - to_calc_length, 1)
    mark_num = len(attrs)
    for i in range(start, n):
        step = 1
        if cnts[i] < 0:
            step = -1
        for j in range(0, mark_num):
            if cnts[i] == 1 or cnts[i] == -1:
                bases[j][i] = attrs[j][i]
                avgs[j][i] = avgs[j][i]
                maxx[j][i] = attrs[j][i]
                minn[j][i] = attrs[j][i]
                min_cnt[j][i] = 0
                max_cnt[j][i] = 0
            else:
                bases[j][i] = bases[j][i - 1]
                if cnts[i] != 0:
                    avgs[j][i] = avgs[j][i - 1] + ((attrs[j][i] - avgs[j][i - 1]) / np.abs(cnts[i]))
                else:
                    avgs[j][i] = avgs[j][i - 1] + (attrs[j][i] - avgs[j][i - 1])
                if cnts[i] > 0 and attrs[j][i] > maxx[j][i - 1]:
                    max_cnt[j][i] = step
                    maxx[j][i] = attrs[j][i]
                elif cnts[i] < 0 and \
                        ((attrs[j][i] < maxx[j][i - 1] and signs[j] is True) or
                         (attrs[j][i] > maxx[j][i - 1] and signs[j] is False)):
                    max_cnt[j][i] = step
                    maxx[j][i] = attrs[j][i]
                else:
                    max_cnt[j][i] = max_cnt[j][i - 1] + step
                    maxx[j][i] = maxx[j][i - 1]
                if cnts[i] > 0 and attrs[j][i] < minn[j][i - 1]:
                    min_cnt[j][i] = step
                    minn[j][i] = attrs[j][i]
                elif cnts[i] < 0 and \
                        ((attrs[j][i] > minn[j][i - 1] and signs[j] is True) or
                         (attrs[j][i] < minn[j][i - 1] and signs[j] is False)):
                    min_cnt[j][i] = step
                    minn[j][i] = attrs[j][i]
                else:
                    min_cnt[j][i] = min_cnt[j][i - 1] + step
                    minn[j][i] = minn[j][i - 1]


@nb.jit(nopython=True)
def _mark_circle_avgs(cnts: np.ndarray, attrs: [], avgs: [], to_calc_length: int):
    n = len(cnts)
    start = max(n - to_calc_length, 1)
    mark_num = len(attrs)
    for i in range(start, n):
        for j in range(0, mark_num):
            if cnts[i] == 1 or cnts[i] == -1:
                avgs[j][i] = avgs[j][i]
            else:
                avgs[j][i] = avgs[j][i - 1] + ((attrs[j][i] - avgs[j][i - 1]) / np.abs(cnts[i]))


@nb.jit(nopython=True)
def _mark_circle_vertexs2(cnt: np.ndarray, attrs: [], signs: np.ndarray, _bases: [],
                          _maxxs: [], _minns: [],
                          # _maxx_ratios: [], _minn_ratios: [],
                          _max_cnts: [], _min_cnts: [], _max_cnt_ratios: [], _min_cnt_ratios: [],
                          _cnts: np.ndarray, steps: list[int], L: int, to_calc_length: int):
    n = len(cnt)
    start = max(n - to_calc_length, 1)
    mark_num = len(attrs)
    step_num = len(steps)
    circle_num = 0
    for s in steps:
        if s > circle_num:
            circle_num = s
    safe_circle_num = max(circle_num+1, 1)
    buf_total_len = safe_circle_num * 2
    # 使用 prange 对不同特征并行处理
    for j in nb.prange(mark_num):
        sign_val = signs[j]
        attr_vec = attrs[j]

        # buffer 形状: [circle_num, 2] -> 0 为 U, 1 为 D
        buf_bases = np.full(buf_total_len, np.nan)
        buf_cnts = np.full(buf_total_len, np.nan)
        buf_maxx = np.full(buf_total_len, np.nan)
        buf_minn = np.full(buf_total_len, np.nan)
        buf_max_cnts = np.full(buf_total_len, np.nan)
        buf_min_cnts = np.full(buf_total_len, np.nan)
        ptr_u_buf = buf_total_len - 2
        ptr_d_buf = buf_total_len - 1

        # 填充历史周期数据
        found_u = 0
        found_d = 0
        for p in range(start - 2, 0, -1):
            if found_u >= safe_circle_num and found_d >= safe_circle_num:
                break
            if cnt[p + 1] == 1 and found_u < safe_circle_num:
                write_idx = (safe_circle_num - 1 - found_u) * 2
                idx0_u = (j * step_num * 2)
                buf_bases[write_idx] = _bases[idx0_u][p]
                buf_cnts[write_idx] = _cnts[idx0_u][p]
                buf_maxx[write_idx] = _maxxs[idx0_u][p]
                buf_minn[write_idx] = _minns[idx0_u][p]
                buf_max_cnts[write_idx] = _max_cnts[idx0_u][p]
                buf_min_cnts[write_idx] = _min_cnts[idx0_u][p]
                found_u += 1
            elif cnt[p + 1] == -1 and found_d < safe_circle_num:
                write_idx = (safe_circle_num - 1 - found_d) * 2 + 1
                idx0_d = (j * step_num * 2) + 1
                buf_bases[write_idx] = _bases[idx0_d][p]
                buf_cnts[write_idx] = _cnts[idx0_d][p]
                buf_maxx[write_idx] = _maxxs[idx0_d][p]
                buf_minn[write_idx] = _minns[idx0_d][p]
                buf_max_cnts[write_idx] = _max_cnts[idx0_d][p]
                buf_min_cnts[write_idx] = _min_cnts[idx0_d][p]
                found_d += 1

        for i in range(start, n):
            val = attr_vec[i]
            if np.isnan(val):
                continue
            # 使用简单的映射建立 mode (位值)
            # 1: U结算, 2: D结算, 0: 全增量
            mode = 0
            if cnt[i] == 1:
                mode = 1
            elif cnt[i] == -1:
                mode = 2

            for side in range(2):  # 0: U, 1: D
                # 通过位移和与运算判断当前 side 是否需要结算
                # 如果 (mode >> 0) & 1 为真，处理 U 结算
                # 如果 (mode >> 1) & 1 为真，处理 D 结算
                is_settle = (mode >> side) & 1
                is_rev = (side == 1 and sign_val)

                if is_settle:
                    # --- 路径 A: 结算逻辑 ---
                    if side == 0:
                        ptr_u_buf = (ptr_u_buf + 2) % buf_total_len
                        curr_ptr = ptr_u_buf
                    else:
                        ptr_d_buf = (ptr_d_buf + 2) % buf_total_len
                        curr_ptr = ptr_d_buf

                    for sidx in range(step_num):
                        step = steps[sidx]
                        # 更新具体的列索引 (sidx * 2 是为了跳过 step 的 U/D 对)
                        out_idx = (j * step_num * 2) + (sidx * 2) + side

                        # 1. 存入缓冲区
                        if step == 0 and not np.isnan(_cnts[out_idx][i - 1]):
                            buf_bases[curr_ptr] = _bases[out_idx][i - 1]
                            buf_cnts[curr_ptr] = _cnts[out_idx][i - 1]
                            buf_maxx[curr_ptr] = _maxxs[out_idx][i - 1]
                            buf_minn[curr_ptr] = _minns[out_idx][i - 1]
                            buf_max_cnts[curr_ptr] = _max_cnts[out_idx][i - 1]
                            buf_min_cnts[curr_ptr] = _min_cnts[out_idx][i - 1]

                        # 2. 初始化新起点
                        _bases[out_idx][i] = val
                        _cnts[out_idx][i] = 1
                        _maxxs[out_idx][i] = val
                        _minns[out_idx][i] = val
                        _max_cnts[out_idx][i] = 1
                        _min_cnts[out_idx][i] = 1

                        # 3. 回溯
                        for sp in range(0, step):
                            look_idx = (curr_ptr - sp * 2) % buf_total_len
                            if np.isnan(buf_maxx[look_idx]): break
                            b_mx, b_mn = buf_maxx[look_idx], buf_minn[look_idx]
                            # 极值合并
                            if (not is_rev and b_mx > _maxxs[out_idx][i]) or (is_rev and b_mx < _maxxs[out_idx][i]):
                                _maxxs[out_idx][i] = b_mx
                                _max_cnts[out_idx][i] = _cnts[out_idx][i] + buf_max_cnts[look_idx]
                            if (not is_rev and b_mn < _minns[out_idx][i]) or (is_rev and b_mn > _minns[out_idx][i]):
                                _minns[out_idx][i] = b_mn
                                _min_cnts[out_idx][i] = _cnts[out_idx][i] + buf_min_cnts[look_idx]
                            _cnts[out_idx][i] += buf_cnts[look_idx]
                            _bases[out_idx][i] = buf_bases[look_idx]

                        _min_cnt_ratios[out_idx][i] = (_min_cnts[out_idx][i] + L) / (_cnts[out_idx][i] + (L * 2))
                        _max_cnt_ratios[out_idx][i] = (_max_cnts[out_idx][i] + L) / (_cnts[out_idx][i] + (L * 2))
                else:
                    # --- 路径 B: 增量逻辑 ---
                    for sidx in range(step_num):
                        out_idx = (j * step_num * 2) + (sidx * 2) + side
                        if np.isnan(_cnts[out_idx][i - 1]):
                            _bases[out_idx][i] = val
                            _cnts[out_idx][i] = 1
                            _maxxs[out_idx][i] = val
                            _minns[out_idx][i] = val
                            _max_cnts[out_idx][i] = 1
                            _min_cnts[out_idx][i] = 1
                            continue
                        _bases[out_idx][i] = _bases[out_idx][i - 1]
                        _cnts[out_idx][i] = _cnts[out_idx][i - 1] + 1
                        # Maxx 增量
                        if (not is_rev and val > _maxxs[out_idx][i - 1]) or (is_rev and val < _maxxs[out_idx][i - 1]):
                            _maxxs[out_idx][i] = val
                            _max_cnts[out_idx][i] = 1
                        else:
                            _maxxs[out_idx][i] = _maxxs[out_idx][i - 1]
                            _max_cnts[out_idx][i] = _max_cnts[out_idx][i - 1] + 1  # 引用保持一致
                        # Minn 增量
                        if (not is_rev and val < _minns[out_idx][i - 1]) or (is_rev and val > _minns[out_idx][i - 1]):
                            _minns[out_idx][i] = val
                            _min_cnts[out_idx][i] = 1
                        else:
                            _minns[out_idx][i] = _minns[out_idx][i - 1]
                            _min_cnts[out_idx][i] = _min_cnts[out_idx][i - 1] + 1
                        _min_cnt_ratios[out_idx][i] = (_min_cnts[out_idx][i] + L) / (_cnts[out_idx][i] + (L * 2))
                        _max_cnt_ratios[out_idx][i] = (_max_cnts[out_idx][i] + L) / (_cnts[out_idx][i] + (L * 2))


class OhlcHandler:
    sorted_indices: np.ndarray
    event_prices: np.ndarray
    event_types: np.ndarray
    event_values: np.ndarray
    total_qty: np.float64


def _numba_cumulative_quantities_ohlc_handled(
        handler: OhlcHandler,
        target_ratio: float
) -> np.float64:
    target = handler.total_qty * target_ratio
    # --- Process Events to Find the Target Price ---
    cum_qty = 0.0
    current_density = 0.0
    # Get the price of the very first event to initialize
    first_event_idx = handler.sorted_indices[0]
    last_price = handler.event_prices[first_event_idx]
    num_events = len(handler.sorted_indices)

    found = False
    for j in range(num_events):
        event_idx = handler.sorted_indices[j]
        price = handler.event_prices[event_idx]
        etype = handler.event_types[event_idx]
        value = handler.event_values[event_idx]

        # Calculate quantity accumulated in the price interval since the last event
        price_delta = price - last_price
        if current_density > 0 and price_delta > 0:
            qty_in_interval = current_density * price_delta

            # Check if the target is met within this continuous interval
            if cum_qty < target and cum_qty + qty_in_interval >= target:
                price_needed = (target - cum_qty) / current_density
                return last_price + price_needed

            cum_qty += qty_in_interval

        # Process the event itself at the current price point.
        # Unstable sort order for same-priced events is fine because price_delta is 0.
        if etype == 2:  # Point event
            # Check if the target is met by this point quantity
            if cum_qty < target and cum_qty + value >= target:
                return price
            cum_qty += value
        elif etype == 0:  # Start event
            current_density += value
        elif etype == 1:  # End event
            current_density -= value

        last_price = price

    return np.nan


@nb.jit(nopython=True)
def _numba_cumulative_quantities_ohlc_create_handler(
        price_open,
        price_close,
        quantities,
        timeperiod=144,
        cnt=1
) -> list[OhlcHandler]:
    nn = int((timeperiod + 1) * 2.3)
    vector = np.ones(nn)
    if nn > 1:
        for i in range(nn - 1):
            vector[nn - 2 - i] = (vector[nn - 1 - i]) * (timeperiod - 1) / (timeperiod + 1)

    n = len(price_open)

    # Pre-allocate arrays for events to avoid allocation inside the loop,
    # which is faster in Numba. Maximum possible events for a window is 2 * timeperiod.

    handlers = OhlcHandler[cnt]
    for i in range(n - cnt, n):
        # Use a standard sliding window implementation.
        # The window size is `min(i + 1, timeperiod)`.
        start = max(0, i - nn + 1)
        size = i - start + 1

        window_open = price_open[start:i + 1]
        window_close = price_close[start:i + 1]

        # Apply the weighting vector to the quantities in the window.
        window_quantities = quantities[start:i + 1] * vector[nn - size:] * 2

        handler = OhlcHandler()
        max_events = 2 * nn
        event_prices = np.empty(max_events)
        # Event types: 0=start of a range, 1=end of a range, 2=point quantity
        event_types = np.empty(max_events, dtype=np.uint8)
        event_values = np.empty(max_events)

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

            handler.total_qty = window_quantities.sum()
            handler.event_prices = event_prices
            handler.event_types = event_prices
            handler.event_values = event_prices
            handlers[k + cnt - n] = handler

        return handlers


@nb.jit(nopython=True)
def _numba_cumulative_quantities_ohlc(
        price_open,
        price_close,
        quantities,
        target_ratio,
        timeperiod,
        to_calc_length
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
    if to_calc_length == 0:
        to_calc_length = len(price_open)
    # The weighting vector calculation remains the same.
    # This creates a geometrically decaying weight.
    nn = timeperiod
    # int((timeperiod + 1) * 2.3)
    vector = np.ones(nn)
    if nn > 1:
        for i in range(nn - 1):
            vector[nn - 2 - i] = (vector[nn - 1 - i]) * (timeperiod - 1) / (timeperiod + 1)

    n = len(price_open)
    result = np.empty(n)

    # Pre-allocate arrays for events to avoid allocation inside the loop,
    # which is faster in Numba. Maximum possible events for a window is 2 * timeperiod.
    max_events = 2 * nn
    event_prices = np.empty(max_events)
    # Event types: 0=start of a range, 1=end of a range, 2=point quantity
    event_types = np.empty(max_events, dtype=np.uint8)
    event_values = np.empty(max_events)

    n_start = max(n - to_calc_length, 0)
    for i in range(n_start, n):
        # Use a standard sliding window implementation.
        # The window size is `min(i + 1, timeperiod)`.
        start = max(0, i - nn + 1)
        size = i - start + 1

        window_open = price_open[start:i + 1]
        window_close = price_close[start:i + 1]

        # Apply the weighting vector to the quantities in the window.
        window_quantities = quantities[start:i + 1] * vector[nn - size:] * 2

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
        timeperiod=144,
        to_calc_length=0
) -> np.ndarray:
    ver = f"_ratio_{target_ratio}_timeperiod_{timeperiod}_0001"
    src = f"open_{price_open.name}_close_{price_close.name}_quantities_{quantities.name}"
    data = _auto_load(config, "numba_cumulative_quantities_ohlc", src, ver)
    if data is None:
        data = _numba_cumulative_quantities_ohlc(price_open.values, price_close.values, quantities.values,
                                                 target_ratio, timeperiod, to_calc_length)
        _auto_save(config, data, "numba_cumulative_quantities_ohlc", src, ver)
    return data


def numba_cumulative_quantities_ohlc_create_handler(
        config: dict,
        price_open: pd.Series,
        price_close: pd.Series,
        quantities: pd.Series,
        target_ratio=0.1,
        timeperiod=144,
        cnt=1,
) -> list[OhlcHandler]:
    return _numba_cumulative_quantities_ohlc_create_handler(price_open.values, price_close.values, quantities.values,
                                                            timeperiod, cnt)


def numba_cumulative_quantities_ohlc_handled(
        config: dict,
        handlers: list[OhlcHandler],
        target_ratio=0.1) -> np.ndarray:
    cnt = len(handlers)
    results = np.empty(len(handlers))
    for i in range(0, cnt):
        results[i] = _numba_cumulative_quantities_ohlc_handled(handlers[i], target_ratio)


def mark_flines_numba(config: dict, df: DataFrame, threshold=5.0, col_num=9):
    ver = f"threshold_{threshold}_col_num_{col_num}_0001"
    n = len(df['date'])
    fline_cols = np.empty((col_num, n), dtype=np.float64)
    fline_date_cols = np.empty((col_num, n), dtype=np.float64)
    fullloaded = True
    # print("mark_flines_numba")
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


@njit
def _rolling_rank_inplace_kernel(values, target, window, to_calc_length):
    """
    values: 输入数据数组
    target: 输出结果数组 (视图)
    window: 窗口大小
    to_calc_length: 需要更新的末尾长度
    """
    n = len(values)
    # 计算起始位置，确保不越界
    start_pos = max(window - 1, n - to_calc_length)

    for i in range(start_pos, n):
        # 获取当前窗口
        # 注意：Numba 切片不产生副本，性能极高
        current_val = values[i]
        count = 0
        # 手动计算排名 (Percent Rank 逻辑)
        for j in range(i - window + 1, i + 1):
            if values[j] <= current_val:
                count += 1
        target[i] = (count / window) * 100


def calc_rank(config: dict, df: pd.DataFrame, attr: str, window: int, rank_attr: str, to_calc_length: int):
    ver = f"{attr}_{window}_0001"
    # 1. 尝试从本地加载缓存数据
    cached_rank = _auto_load(config, "calc_rank", "rank", ver)

    # 2. 防御性初始化：确保列已在内存中“占坑”
    # 这是保证外部指针（如 Java 端或监控模块）能读到新列的关键
    if rank_attr not in df.columns:
        df[rank_attr] = np.nan

    # 获取指向 df 内部内存的视图
    target_view = df[rank_attr].values
    n = len(df)

    if cached_rank is not None:
        # --- 场景 A: 存在缓存 ---
        # 必须使用 [:] 原地填充，严禁直接 df[rank_attr] = cached_rank
        # 长度校验，防止缓存与当前 df 长度不符
        fill_len = min(len(cached_rank), n)
        target_view[-fill_len:] = cached_rank[-fill_len:]
    else:
        # --- 场景 B: 需要计算 ---
        input_values = df[attr].values

        if to_calc_length >= n:
            # 全量计算：先重置内存内容
            target_view[:] = np.nan
            _rolling_rank_inplace_kernel(input_values, target_view, window, n)
        else:
            # 增量计算：只修改末尾部分，前面保持不变
            _rolling_rank_inplace_kernel(input_values, target_view, window, to_calc_length)

        # 保存计算结果（保存整个数组或当前状态）
        # 注意：这里传的是副本，防止异步保存时原始 df 被修改
        _auto_save(config, target_view.copy(), "calc_rank", "rank", ver)


def mark_boundary_numba(config: dict, df: DataFrame, to_calc_length, threshold=5.0):
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
        dates = df['date'].astype('int64').values

        if to_calc_length >= n:
            # 预分配结果数组 (假设已经预先创建了这些列)
            bound_date = np.zeros(n)
            fline_col = np.zeros(n)
            prev_fline_col = np.zeros(n)
            prev_prev_fline_col = np.zeros(n)
            direction = np.zeros(n)
            prev_direction = np.zeros(n)
        else:
            bound_date = df['bound-date'].values
            fline_col = df['bound-line'].values
            prev_fline_col = df['bound-prev-line'].values
            prev_prev_fline_col = df['bound-prev-prev-line'].values
            direction = df['bound-direction'].values
            prev_direction = df['bound-prev-direction'].values

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

    # print(prev_fline_col)
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


def calc_trend_ratio(config: dict, attr: str, data: np.ndarray, ratio: np.ndarray,
                     wavelenth: int, multi: int, to_calc_length: int) -> np.ndarray:
    ver = f"wavelenth_{wavelenth}_multi_{multi}_v_0001"
    # ver = f"wavelenth_{wavelenth}_multi_{multi}_v_0002sqrt"
    load = _auto_load(config, "calc_trend_ratio", attr, ver)
    if load is not None:
        np.copyto(ratio, load)
        return load
    src = data
    n = len(src)
    dest = ratio
    start = max(wavelenth * (multi + 1), n - to_calc_length)
    for i in range(start, n):
        # 切片提取从i开始的k+1个元素
        sub_src = src[np.arange(i - wavelenth * (multi + 1), i + 1, wavelenth)]  # i, i+j, i+2j, ..., i+k
        diffs = sub_src[:-1] - sub_src[1:]
        mask = sub_src[:-1] < sub_src[1:]
        # 计算满足前一个元素小于后一个元素的次数总和
        negative = np.sum(diffs[mask])
        positive = np.sum(diffs) - negative
        dest[i] = np.sqrt((src[i] - src[i - (wavelenth * multi)]) * (src[i] - src[i - (wavelenth * multi)]) * 100 \
                          / src[i] / (positive - negative)) * (1 if src[i] > src[i - (wavelenth * multi)] else -1)
    _auto_save(config, dest, "calc_trend_ratio", attr, ver)
    return dest


def calc_trend_ratio_ema(config: dict, dataframe: DataFrame, attr: str, dst: np.ndarray,
                         wavelenth: int, multi: int, timeperoid: int, to_calc_length=0) -> np.ndarray:
    ver = f"wavelenth_{wavelenth}_multi_{multi}_timeperoid_{timeperoid}_v_0001"  # 0001：没开平方版本
    # ver = f"wavelenth_{wavelenth}_multi_{multi}_timeperoid_{timeperoid}_v_0002sqrt"\
    if to_calc_length == 0:
        to_calc_length = len(dataframe)
    load = _auto_load(config, "calc_trend_ratio_ema", attr, ver)
    if load is not None:
        return load
    n = len(dataframe)
    data = dataframe[attr].values
    if to_calc_length >= n:
        ratio = np.zeros(n)
    else:
        ratio = dataframe[f'tr-{attr}-{wavelenth}-{multi}-{timeperoid}'].values
    calc_trend_ratio(config, attr, data, ratio, wavelenth, multi, to_calc_length)
    dataframe[f'tr-{attr}-{wavelenth}-{multi}-{timeperoid}'] = ratio
    # print(f'calc_trend_ratio_ema1:dst = ')
    # print(dst)
    if to_calc_length >= n:
        dst = ta.EMA(ratio, timeperiod=timeperoid)
    else:
        # print(f'calc_trend_ratio_ema2:ratio = tocal:{to_calc_length}, timeperoid:{timeperoid}')
        # print(ratio[-to_calc_length - timeperoid:])
        ema = ta.EMA(ratio[-to_calc_length - timeperoid:], timeperiod=timeperoid)
        # print(ema)
        np.copyto(dst[-to_calc_length:], ema[-to_calc_length:])
        # print(dst[-to_calc_length:])
    _auto_save(config, dst, "calc_trend_ratio_ema", attr, ver)
    return dst


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
            hurst_rs[i] = np.nan
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


def gaussian(src: np.ndarray, dst: np.ndarray, lookback: int, startAtBar: int, to_calc_length: int):
    """
    vectorized calculate for gaussian curve
    :param src:
    :param lookback:
    :param startAtBar:
    :return:
    """
    n = len(src)
    csrc = src
    if to_calc_length < n:
        nl = to_calc_length + startAtBar + 2
        csrc = src.loc[-nl:].copy()
    currentWeight = [0.0] * len(csrc)
    cumulativeWeight = 0.0
    for i in range(startAtBar + 2):
        y = csrc.shift(i, fill_value=0.0)
        w = math.exp(-(i ** 2) / (2 * lookback ** 2))
        currentWeight += y.values * w
        cumulativeWeight += w
    cdst = currentWeight / cumulativeWeight
    cdst[:startAtBar + 1] = 0.0

    if to_calc_length >= n:
        np.copyto(dst, cdst)
    else:
        dst[-to_calc_length:] = cdst[-to_calc_length:]

    return dst


def calc_quantile(df: pd.DataFrame, wsconfig: dict, to_calc_length: int, attr: str, window: int, quantile: np.float64,
                  df_attr: str = None) -> np.ndarray:
    ver = f"window{window}_0004"
    key = 'median' if quantile == 0.5 else 'q' + str(quantile).replace('.', '_')

    # 1. 确定特征名称
    feature_name = df_attr if df_attr else f'_{attr}-quantile-{window}-{key}'

    # 2. 防御性初始化：确保列在内存中占坑
    if feature_name not in df.columns:
        df[feature_name] = np.nan

    # 获取原始内存视图，保证外部指针同步
    target_view = df[feature_name].values
    n = len(df)

    # 3. 加载缓存 (此处假设 _auto_load 逻辑开启)
    data = _auto_load(wsconfig, f"quantile_{attr}", f'{key}', ver)

    if data is not None:
        fill_len = min(len(data), n)
        target_view[-fill_len:] = data[-fill_len:]
    else:
        # 4. 核心计算逻辑
        if to_calc_length >= n:
            # --- 全量计算 ---
            if quantile == 0.5:
                res = df[attr].rolling(window=window).median()
            else:
                res = df[attr].rolling(window=window).quantile(quantile)
            # 必须使用 [:] 保证原地覆盖，不丢失物理地址
            target_view[:] = res.values
        else:
            # --- 增量计算 ---
            # 仅计算末尾必要的长度，减少计算量
            m = to_calc_length
            # 为了计算最后 m 个点的分位数，需要前溯 window-1 个点
            calc_slice = df[attr].iloc[-(m + window - 1):]

            if quantile == 0.5:
                res = calc_slice.rolling(window=window).median()
            else:
                res = calc_slice.rolling(window=window).quantile(quantile)

            # 只更新末尾 m 个值，保持前面的数据不动
            target_view[-m:] = res.values[-m:]

        # 5. 统一保存：保存的是已经修改好的数组
        _auto_save(wsconfig, target_view, f"quantile_{attr}", f'{key}', ver)

    # 统一返回 numpy 数组视图
    return target_view


def calculate_hma(df: DataFrame, wsconfig: dict, to_calc_length: int, attr: str, timeperiod: int) -> pd.Series:
    """
    计算霍尔移动平均 (Hull Moving Average, HMA)。

    HMA = WMA(2 * WMA(n/2) - WMA(n)), sqrt(n))

    参数:
        price_series: 输入的价格序列 (例如 'close' 列)。
        period: HMA 的计算周期 n。

    返回:
        pandas.Series: 包含 HMA 值的序列。
    """
    price_series = df[attr]
    if timeperiod <= 1:
        return price_series.copy()
    half_period = timeperiod // 2
    sqrt_period = int(math.sqrt(timeperiod))
    wma1 = ta.WMA(price_series, timeperiod=half_period)
    wma2 = ta.WMA(price_series, timeperiod=timeperiod)
    hma_raw = 2 * wma1 - wma2
    final_smoothing_period = max(1, sqrt_period)
    hma_final = ta.WMA(hma_raw, timeperiod=final_smoothing_period)
    return hma_final


def calculate_long_mean(df: DataFrame, wsconfig: dict, to_calc_length: int,
                        attr: str, timeperiod: int, init_value: np.float64, mean_attr: str = None) -> DataFrame:
    n = len(df)
    if mean_attr is None:
        mean_attr = f'{attr}-mean-{timeperiod}'
    if mean_attr not in df.columns:
        df[mean_attr] = np.nan
    mean = df[mean_attr].values
    value = df[attr].values
    _calc_mean(value, mean, init_value, timeperiod, to_calc_length)


def mark_hit_band(df: DataFrame, wsconfig: dict, to_calc_length: int,
                  attr_h: str, attr_l: str, close: str, step: int = 5, step_max: int = 20,
                  tgt_attr: str = None, tgt_cnt_attr: str = None):
    if tgt_attr is None:
        tgt_attr = f'{attr_h}-bhit'
    if tgt_cnt_attr is None:
        tgt_cnt_attr = f'{tgt_attr}-cnt'
    if tgt_attr not in df.columns:
        df[tgt_attr] = np.nan
    if tgt_cnt_attr not in df.columns:
        df[tgt_cnt_attr] = 0
    hit = df[tgt_attr].values
    hitcnt = df[tgt_cnt_attr].values
    _mark_hit_band(df[attr_h].values, df[attr_l].values, df[close].values, hit, hitcnt, step, step_max, to_calc_length)


def mark_hit_line(df: DataFrame, wsconfig: dict, to_calc_length: int,
                  attr: str, close: str, step_max: int = 30, tgt_attr: str = None):
    if tgt_attr is None:
        tgt_attr = f'{attr}-hit'
    if tgt_attr not in df.columns:
        df[tgt_attr] = 0
    hit = df[tgt_attr].values
    _mark_hit_line(df[attr].values, df[close].values, hit, step_max, to_calc_length)
    df[tgt_attr] = hit


def mark_circle_base(df: DataFrame, wsconfig: dict, to_calc_length: int, side_attr: str,
                     idx_attr: str = None, cnt_attr: str = None):
    if idx_attr is None:
        idx_attr = side_attr.replace('side', 'idx')
    if cnt_attr is None:
        cnt_attr = side_attr.replace('side', 'cnt')
    if idx_attr not in df.columns:
        df[idx_attr] = 0
    if cnt_attr not in df.columns:
        df[cnt_attr] = 0
    idx = df[idx_attr].values
    cnt = df[cnt_attr].values
    _mark_circle_base(df[side_attr].values, idx, cnt, to_calc_length)


# chapter系列的生成
def mark_circle_vertexs(df: DataFrame, wsconfig: dict, to_calc_length: int, cnt_attr: str, mark_attrs: [],
                        sides: [], mark_attr_new_names: []):
    n = len(df)
    num = len(mark_attrs)
    _cnt = df[cnt_attr].values.astype('float32')
    _attrs = []
    _bases = []
    _avgs = []
    _maxx = []
    _minn = []
    _max_cnts = []
    _min_cnts = []
    if to_calc_length >= n:
        attrs = []
        for j in range(0, num):
            attrs.extend([f'{mark_attr_new_names[j]}-base', f'{mark_attr_new_names[j]}-avg',
                          f'{mark_attr_new_names[j]}-maxx', f'{mark_attr_new_names[j]}-minn',
                          f'{mark_attr_new_names[j]}-maxxcnt', f'{mark_attr_new_names[j]}-minncnt'])
        df.loc[:, attrs] = np.nan
        df[attrs] = df[attrs].astype('float32')
    for j in range(0, num):
        _attrs.append(df[mark_attrs[j]].values.astype('float32'))
        _bases.append(df[f'{mark_attr_new_names[j]}-base'].values)
        _avgs.append(df[f'{mark_attr_new_names[j]}-avg'].values)
        _maxx.append(df[f'{mark_attr_new_names[j]}-maxx'].values)
        _minn.append(df[f'{mark_attr_new_names[j]}-minn'].values)
        _max_cnts.append(df[f'{mark_attr_new_names[j]}-maxxcnt'].values)
        _min_cnts.append(df[f'{mark_attr_new_names[j]}-minncnt'].values)
    _mark_circle_vertexs(_cnt, _attrs, sides, _bases, _avgs, _maxx, _max_cnts, _minn, _min_cnts, to_calc_length)


# 2x思路下系列的生成
def mark_circle_vertexs2(df: DataFrame, wsconfig: dict, to_calc_length: int, cnt_attr: str,
                         mark_attrs: [], sides: [], mark_attr_new_names: [], steps: list[int], L: int):
    n = len(df)
    num = len(mark_attrs)
    side_hints = ['U', 'D']

    steps = sorted(list(set(steps) | {0}))
    step_num = len(steps)

    matrix_keys = ['base', 'cnt', 'maxx', 'minn', 'maxxcnt', 'minncnt']
    ratio_keys = ['maxxcntratio', 'minncntratio']
    all_keys = matrix_keys + ratio_keys

    # --- 策略 A: 冷启动 (一次性创建并建立连续视图) ---
    if to_calc_length >= n:
        # 这个分支只在系统刚启动，或者需要全量重算时走一次
        for key in all_keys:
            total_rows = num * step_num * 2
            big_matrix = np.full((total_rows, n), np.nan, dtype=np.float32)

            current_batch_cols = []
            cursor = 0
            for j in range(num):
                for step in steps:
                    for side_h in side_hints:
                        col_name = f'{side_h}{step}#{mark_attr_new_names[j]}-{key}'
                        current_batch_cols.append(col_name)
                        cursor += 1

            df[current_batch_cols] = big_matrix.T

    # --- 策略 B: 热运行 (处理动态增长的数据列) ---
    # 即使列已存在且行数增加了，我们也要提取出最后一段的连续视图传给 JIT
    def get_raw_mem_list(suffix):
        raw_arrays = []
        for j in range(num):
            for step in steps:
                for side_h in side_hints:
                    col = f'{side_h}{step}#{mark_attr_new_names[j]}-{suffix}'
                    # 获取底层 numpy 数组的视图
                    raw_arrays.append(df[col].values)
        return raw_arrays

    # 提取输入（只取需要计算的末尾部分）
    _cnt = df[cnt_attr].values.astype('float32')
    _exist_attrs = df.columns.values.tolist()
    print(set(_exist_attrs) & set(mark_attrs))
    _attrs = [df[m].values.astype('float32') for m in mark_attrs if m in _exist_attrs]
    _sides = np.array(sides, dtype=np.float32)
    _steps = np.array(steps)

    # 提取所有工作矩阵
    m_args = {key: get_raw_mem_list(key) for key in all_keys}

    _mark_circle_vertexs2(
        _cnt, _attrs, _sides,
        m_args['base'], m_args['maxx'], m_args['minn'],
        # m_args['maxxratio'], m_args['minnratio'],
        m_args['maxxcnt'], m_args['minncnt'],  # 注意 JIT 签名里的命名对应
        m_args['maxxcntratio'], m_args['minncntratio'],
        m_args['cnt'], _steps, L, to_calc_length
    )
    # --- 关键：热运行下的数据同步回填 ---
    # 如果 to_calc_length < n，说明我们是在一个副本切片上做的计算
    # 必须把计算结果写回 df 的末尾
    if to_calc_length < n:
        start_idx = n - to_calc_length
        for key in all_keys:
            matrix = m_args[key]
            cursor = 0
            for j in range(num):
                for step in steps:
                    for side_h in side_hints:
                        col = f'{side_h}{step}#{mark_attr_new_names[j]}-{key}'
                        # 只写回末尾计算的部分
                        df[col].values[start_idx:] = matrix[cursor, start_idx:]
                        cursor += 1


def mark_circle_avgs(df: DataFrame, wsconfig: dict, to_calc_length: int, cnt_attr: str, mark_attrs: [],
                     mark_attr_new_names: []):
    n = len(df)
    num = len(mark_attrs)
    _cnt = df[cnt_attr].values
    _attrs = []
    _avgs = []
    for j in range(0, num):
        _attrs.append(df[mark_attrs[j]].values)
        if to_calc_length >= n:
            _avgs.append(np.zeros(n))
        else:
            _avgs.append(df[f'{mark_attr_new_names[j]}-avg'])
    _mark_circle_avgs(_cnt, _attrs, _avgs, to_calc_length)
    for j in range(0, num):
        df[f'{mark_attr_new_names[j]}-avg'] = _avgs[j]


##同一化接口
def calculate_rank(df: DataFrame, wsconfig: dict, to_calc_length: int, attr: str, window: int,
                   rank_attr: str = None) -> DataFrame:
    if rank_attr is None:
        rank_attr = f'{attr}-rank-{window}'
    calc_rank(wsconfig, df, attr, window, rank_attr, to_calc_length)


def calculate_dstr(df: DataFrame, wsconfig: dict, to_calc_length: int,
                   window, ratio: np.float64 = 0.5,
                   open_attr: str = 'open', close_attr: str = 'close', volume_attr: str = 'volume'):
    return numba_cumulative_quantities_ohlc(wsconfig,
                                            df[open_attr], df[close_attr], df[volume_attr],
                                            target_ratio=ratio, timeperiod=window,
                                            to_calc_length=to_calc_length)


def calculate_quantile(df: DataFrame, wsconfig: dict, to_calc_length: int, attr: str, window: int, quantile: np.float64,
                       quantile_attr: str):
    calc_quantile(df, wsconfig, to_calc_length, attr, window, quantile, df_attr=quantile_attr)


def calculate_iqrscore(df: DataFrame, wsconfig: dict, to_calc_length: int, attr: str, window: int,
                       iqr_attr: str = None, median_attr: str = None):
    if iqr_attr is None:
        iqr_attr = f'{attr}-iqrscore-{window}'

    # 1. 确保列占坑，保持外部指针同步
    if iqr_attr not in df.columns:
        df[iqr_attr] = np.nan

    # 2. 调用重构后的 calc_quantile (内部已处理好原地更新)
    q3 = calc_quantile(df, wsconfig, to_calc_length, attr, window, 0.75)
    q1 = calc_quantile(df, wsconfig, to_calc_length, attr, window, 0.25)
    media = calc_quantile(df, wsconfig, to_calc_length, attr, window, 0.5, df_attr=median_attr)

    target_view = df[iqr_attr].values
    val_view = df[attr].values
    n = len(df)
    m = to_calc_length

    # 3. 执行计算逻辑
    if m >= n:
        # 全量计算
        diff = q3 - q1
        # 防御性处理：防止除以 0
        target_view[:] = np.where(diff != 0, (val_view - media) / diff, 0.0)
    else:
        # 增量计算：注意使用 [-m:] 切片而非 [-m] 单索引
        v_part = val_view[-m:]
        m_part = media[-m:]
        diff_part = q3[-m:] - q1[-m:]

        # 原地更新最后 m 个点
        target_view[-m:] = np.where(diff_part != 0, (v_part - m_part) / diff_part, 0.0)



@nb.njit
def _calculate_qnrscore(_attr: np.ndarray, _qns: np.ndarray, window: int, to_calc_length: int):
    n = len(_attr)
    start = max(n - to_calc_length, window - 1)
    for i in range(start, n):
        # 定义当前窗口
        window_data = _attr[i - window + 1: i + 1]
        current_value = _attr[i]
        rank_count = 0
        valid_count = 0
        for val in window_data:
            if not np.isnan(val):
                valid_count += 1
                if val <= current_value:
                    rank_count += 1
        if valid_count > 0:
            percentile_rank = rank_count / valid_count
            _qns[i] = percentile_rank - 0.5
        else:
            _qns[i] = np.nan


@nb.njit
def pure_numba_searchsorted(arr: List, val):
    """
    纯 Numba 实现的二分查找，用于查找 val 的插入位置 (side='left')
    """
    low = 0
    high = len(arr)

    while low < high:
        mid = (low + high) // 2
        # 注意：这里直接对 List 进行索引，避免了 np.asarray()
        if arr[mid] < val:
            low = mid + 1
        else:
            high = mid

    return low  # low 就是 side='left' 的插入索引


@nb.njit
def binary_search_remove(arr: List, val):
    """
    在有序 List 中查找并移除第一个匹配的 val。
    如果找到，返回 True；否则返回 False。
    """
    # Numba 不支持 List.remove()，所以我们手动查找并删除
    n = len(arr)

    # 使用 searchsorted 找到可能的插入点，然后检查周围元素
    idx = pure_numba_searchsorted(arr, val)

    if idx < n and arr[idx] == val:
        # 找到匹配，执行删除
        for j in range(idx, n - 1):
            arr[j] = arr[j + 1]
        arr.pop()
        return True
    return False


@nb.njit
def binary_search_insert(arr: List, val):
    """
    在有序 List 中插入 val 以保持其排序。
    """
    n = len(arr)
    if n == 0:
        arr.append(val)
        return

    # 在 List 上使用 np.searchsorted 需要先转换成 np.ndarray
    idx = pure_numba_searchsorted(arr, val)

    # 手动执行 List 插入操作 (Numba list的插入效率比Python list高)
    arr.append(arr[n - 1])  # 先扩展List大小
    for j in range(n - 1, idx, -1):
        arr[j] = arr[j - 1]
    arr[idx] = val


@nb.njit
def bit_update(bit_array: np.ndarray, index: int, delta: int):
    """
    Fenwick Tree 更新操作。
    index 必须是 1-based (即 1 到 M)。
    """
    i = index
    # 循环遍历 BIT 树结构，更新相关节点
    while i < len(bit_array):
        bit_array[i] += delta
        i += i & (-i)  # i += LSB(i)


@nb.njit
def bit_query(bit_array: np.ndarray, index: int):
    """
    Fenwick Tree 查询操作：查询索引 <= index 的累积频率 (即排名)。
    index 必须是 1-based。
    """
    s = 0
    i = index
    # 循环遍历 BIT 树结构，累加相关节点
    while i > 0:
        s += bit_array[i]
        i -= i & (-i)  # i -= LSB(i)
    return s


# 假设 M=10000 是全局常量，并且 _attr 是原始浮点数，_discrete_attr 是预处理的整数
@nb.njit
def _calculate_qnrscore_backtest(_attr: np.ndarray, _discrete_attr: np.ndarray,
                                 _qns: np.ndarray, window: int, to_calc_length: int, M_val: int):
    n = len(_attr)
    start = max(window - 1, n - to_calc_length)

    # Fenwick Tree 数组 (1-based 索引，大小为 M_val + 1)
    # 用于存储窗口内每个离散值的频率
    bit_array = np.zeros(M_val + 1, dtype=np.int32)

    # 跟踪窗口内有效（非NaN）的总数
    valid_count = 0

    # 1. 初始化窗口: 索引 0 到 start - 1
    for i in range(start):
        d_val = _discrete_attr[i]
        if not np.isnan(_attr[i]):  # 使用原始浮点数检查NaN
            bit_update(bit_array, d_val + 1, 1)  # BIT index 1-based
            valid_count += 1

    # 2. 滚动窗口计算
    for i in range(start, n):
        # A. 移除旧值 (i - window 处)
        if i >= window:
            old_attr = _attr[i - window]
            if not np.isnan(old_attr):
                old_d_val = _discrete_attr[i - window]
                bit_update(bit_array, old_d_val + 1, -1)  # 移除
                valid_count -= 1

        # B. 加入新值 (当前值 i 处)
        current_attr = _attr[i]
        current_d_val = _discrete_attr[i]
        is_valid_current = not np.isnan(current_attr)

        if is_valid_current:
            bit_update(bit_array, current_d_val + 1, 1)  # 加入
            valid_count += 1

        # C. 计算百分位
        if valid_count > 0 and is_valid_current:
            # 查询排名: 小于等于 current_d_val 的所有频率之和
            rank_count = bit_query(bit_array, current_d_val + 1)

            percentile_rank = rank_count / valid_count
            _qns[i] = percentile_rank - 0.5
        else:
            _qns[i] = np.nan

    return _qns


@nb.njit
def _calculate_qnrscore_fast(_attr: np.ndarray, _qns: np.ndarray, window: int, to_calc_length: int):
    n = len(_attr)
    start = max(window - 1, n - to_calc_length)

    # 使用 numba.typed.List 来存储有序的窗口数据，便于高效插入和删除
    # 注意：Numba List 的实现细节可能导致其插入/删除并非严格 O(window)，
    # 但相比Python List和纯Numpy数组，它在Numba中通常是最优选择。
    _sorted_window = List.empty_list(nb.float64)

    # 1. 初始化第一个窗口
    for i in range(start - window + 1, start):
        if not np.isnan(_attr[i]):
            # 初始窗口，使用高效的插入排序
            binary_search_insert(_sorted_window, _attr[i])

    # 处理 'start' 索引处的最后一个初始数据
    if not np.isnan(_attr[start - 1]):
        binary_search_insert(_sorted_window, _attr[start - 1])

    # 2. 滚动窗口计算
    for i in range(start, n):
        # A. 移除旧值 (如果存在且有效)
        # 待移除的值在 i - window 处
        if i >= window:
            old_value = _attr[i - window]
            if not np.isnan(old_value):
                binary_search_remove(_sorted_window, old_value)

        # B. 加入新值 (当前值)
        current_value = _attr[i]
        is_valid_current = not np.isnan(current_value)
        if is_valid_current:
            binary_search_insert(_sorted_window, current_value)

        # C. 计算百分位
        valid_count = len(_sorted_window)

        if valid_count > 0 and is_valid_current:
            low = 0
            high = valid_count
            val_check = current_value + 1e-9  # 保证浮点数比较的安全

            # 查找第一个严格大于 current_value 的索引 (即 side='right')
            while low < high:
                mid = (low + high) // 2
                if _sorted_window[mid] <= current_value:
                    low = mid + 1
                else:
                    high = mid

            rank_count = low  # low 现在是小于等于 current_value 的元素个数

            # 您原代码中的百分位排名定义是： (<= current_value 的数量) / (有效数量)
            percentile_rank = rank_count / valid_count
            _qns[i] = percentile_rank - 0.5
        else:
            _qns[i] = np.nan

    return _qns


def calculate_qnrscore(df: DataFrame, wsconfig: dict, to_calc_length: int,
                       attr: str, window: int, qnr_attr: str = None) -> pd.DataFrame:
    if qnr_attr is None:
        qnr_attr = f'{attr}-qnrscore-{window}'
    ver = f"0001_timeperoid_{window}"
    load = None
    # load = _auto_load(wsconfig, "calculate_qnrscore", attr, ver)
    if load is not None:
        df[qnr_attr] = load
        return
    n = len(df)
    if to_calc_length >= n:
        df[qnr_attr] = np.full(n, np.nan, dtype=np.float32)
    _attr = df[attr].values
    _qns = df[qnr_attr].values
    if to_calc_length >= n:
        # 提取非 NaN 的有效数据
        valid_attr = _attr[~np.isnan(_attr)]

        # 计算全局边界
        global_min = np.min(valid_attr)
        global_max = np.max(valid_attr)
        data_range = global_max - global_min
        # 归一化（MinMax Scaling）
        if data_range == 0:
            # 所有有效值都一样，映射为 0
            _scaled_attr = np.where(~np.isnan(_attr), 0.0, np.nan)
        else:
            _scaled_attr = (_attr - global_min) / data_range
        M = 10000  # 推荐的精度值
        _discrete_attr = np.full_like(_attr, -1, dtype=np.int32)  # 使用 -1 或其他值代表 NaN
        # 仅对非 NaN 的值进行离散化
        valid_indices = ~np.isnan(_attr)
        _discrete_attr[valid_indices] = np.floor(_scaled_attr[valid_indices] * M).astype(np.int32)
        _calculate_qnrscore_backtest(
            _attr,
            _discrete_attr,
            _qns,
            window,
            to_calc_length,
            M_val=M
        )
    else:
        _calculate_qnrscore(_attr, _qns, window, to_calc_length)
    _auto_save(wsconfig, _qns, "calculate_qnrscore", attr, ver)


@nb.njit
def _mark_extreme_memory(attrs: [], vertex_attrs: [],
                         mems: [], durations: [], vertexes: [], vertex_mems: [],
                         mems_prev: [], durations_prev: [], vertexes_prev: [], vertex_mems_prev: [],
                         timeperoid: int, extremes: list[np.float32], to_calc_length: int):
    n = len(attrs[0])
    num = len(attrs)
    n_start = max(n - to_calc_length, 0)
    m_cut = np.exp((-3))
    m_0 = np.exp(-1 / timeperoid)
    for j in range(0, num):
        attr = attrs[j]
        mem = mems[j]
        mem_prev = mems_prev[j]
        extreme = extremes[j]
        vertex_attr = vertex_attrs[j]
        if durations is not None:
            duration = durations[j]
            duration_prev = durations_prev[j]
        else:
            duration = None
            duration_prev = None
        if vertexes is not None:
            vertex = vertexes[j]
            vertex_mem = vertex_mems[j]
            vertex_prev = vertexes_prev[j]
            vertex_mem_prev = vertex_mems_prev[j]
        else:
            vertex = None
            vertex_mem = None
            vertex_prev = None
            vertex_mem_prev = None
        for i in range(n_start, n):
            if i == 0:
                mem[i] = 0
                mem_prev[i] = 0
                if duration is not None:
                    duration[i] = 0
                    duration_prev[i] = 0
                if vertex is not None:
                    vertex[i] = 0
                    vertex_mem[i] = 0
                    vertex_prev[i] = 0
                    vertex_mem_prev[i] = 0
            switch = False
            if np.abs(attr[i]) > extreme:
                _sign = np.sign(attr[i])
                if np.abs(mem[i - 1]) < 0.9 or np.sign(mem[i - 1]) != _sign:
                    switch = True
                mem[i] = _sign
                if duration is not None:
                    if np.sign(duration[i - 1]) != _sign:
                        duration[i] = _sign
                    else:
                        duration[i] = duration[i - 1] + _sign
                if vertex is not None:
                    if np.sign(vertex[i - 1]) != _sign:
                        vertex[i] = vertex_attr[i]
                        vertex_mem[i] = vertex_attr[i]
                    elif np.abs(attr[i]) > np.abs(vertex[i - 1]):
                        vertex[i] = vertex_attr[i]
                        vertex_mem[i] = vertex_attr[i]
                    elif np.abs(attr[i]) > np.abs(vertex_mem[i - 1]):
                        vertex[i] = vertex[i - 1]
                        vertex_mem[i] = attr[i]
                    else:
                        vertex[i] = vertex[i - 1]
                        vertex_mem[i] = vertex_mem[i - 1] * m_0
            elif mem[i - 1] != 0:
                mem[i] = mem[i - 1] * m_0
                if duration is not None:
                    duration[i] = duration[i - 1]
                if vertex is not None:
                    vertex[i] = vertex[i - 1]
                    vertex_mem[i] = vertex_mem[i - 1] * m_0
                    if np.abs(vertex_mem[i]) < np.abs(vertex[i]) * m_cut:
                        vertex[i] = 0
                        vertex_mem[i] = 0
                if np.abs(mem[i]) < m_cut:
                    mem[i] = 0
            if switch:
                mem_prev[i] = mem[i - 1] * m_0
                if duration is not None:
                    duration_prev[i] = duration[i - 1]
                if vertex is not None:
                    vertex_prev[i] = vertex[i - 1]
                    vertex_mem_prev[i] = vertex_mem[i - 1] * m_0
                    if np.abs(vertex_mem_prev[i]) < np.abs(vertex_prev[i]) * m_cut:
                        vertex_prev[i] = 0
                        vertex_mem_prev[i] = 0
                if np.abs(mem_prev[i]) < m_cut:
                    mem_prev[i] = 0
            elif mem_prev[i - 1] != 0:
                mem_prev[i] = mem_prev[i - 1] * m_0
                if duration_prev is not None:
                    duration_prev[i] = duration_prev[i - 1]
                if vertex_prev is not None:
                    vertex_prev[i] = vertex_prev[i - 1]
                    vertex_mem_prev[i] = vertex_mem_prev[i - 1] * m_0
                    if np.abs(vertex_mem_prev[i]) < np.abs(vertex_prev[i]) * m_cut:
                        vertex_prev[i] = 0
                        vertex_mem_prev[i] = 0
                if np.abs(mem_prev[i]) < m_cut:
                    mem_prev[i] = 0


def mark_extreme_memory(df: DataFrame, wsconfig: dict, to_calc_length: int,
                        attrs: list[str], vertex_attrs: list[str], n_attrs: list[str], timeperiod: int,
                        extremes: list[np.float64],
                        with_dure=True, with_vertex=True):
    num = len(attrs)
    if num == 0:
        return
    n = len(df)
    _attrs = [None] * num
    _mems = [None] * num
    _mems_prev = [None] * num
    _vertex_attrs = [None] * num
    # 1. 定义需要生成的列后缀
    suffixes = ['-mem', '-mem-prev']
    if with_dure:
        suffixes += ['-duration', '-duration-prev']
    if with_vertex:
        suffixes += ['-vertex', '-vertexmem', '-vertex-prev', '-vertexmem-prev']

    # 2. 防御性初始化：确保所有列拥有【独立】的内存空间
    for n_attr in n_attrs:
        for suffix in suffixes:
            col = n_attr + suffix
            if col not in df.columns:
                # 关键：每一列都要独立初始化，防止内存共享
                df[col] = np.float32(0.0)
    # 3. 准备视图列表 (Views) - 直接指向 df 的物理地址
    # 辅助函数：确保拿到的是 float32 视图，避免副本
    def get_f32_view(name):
        # 如果类型不对，建议在外部统一转换，此处做个防御
        if df[name].dtype != np.float32:
            df[name] = df[name].astype(np.float32)
        return df[name].values

    _attrs = [get_f32_view(a) for a in attrs]
    _v_attrs = [get_f32_view(va) for va in vertex_attrs]

    _mems = [get_f32_view(f'{na}-mem') for na in n_attrs]
    _mems_p = [get_f32_view(f'{na}-mem-prev') for na in n_attrs]

    _durations = [get_f32_view(f'{na}-duration') for na in n_attrs] if with_dure else None
    _durations_p = [get_f32_view(f'{na}-duration-prev') for na in n_attrs] if with_dure else None

    if with_vertex:
        _vertexes = [get_f32_view(f'{na}-vertex') for na in n_attrs]
        _v_mems = [get_f32_view(f'{na}-vertexmem') for na in n_attrs]
        _vertexes_p = [get_f32_view(f'{na}-vertex-prev') for na in n_attrs]
        _v_mems_p = [get_f32_view(f'{na}-vertexmem-prev') for na in n_attrs]
    else:
        _vertexes = _v_mems = _vertexes_p = _v_mems_p = None

    # 4. 全量计算时的重置 (原地修改内容 [:] )
    if to_calc_length >= n:
        for i in range(num):
            _mems[i][:] = 0.0
            _mems_p[i][:] = 0.0
            if with_dure:
                _durations[i][:] = 0.0
                _durations_p[i][:] = 0.0
            if with_vertex:
                _vertexes[i][:] = 0.0
                _v_mems[i][:] = 0.0
                _vertexes_p[i][:] = 0.0
                _v_mems_p[i][:] = 0.0
    # 5. 参数准备
    extremes_f32 = np.array(extremes, dtype=np.float32)

    # 6. 调用 Numba/C++ 内核 (原地修改视图内存)
    _mark_extreme_memory(
        _attrs, _v_attrs,
        _mems, _durations, _vertexes, _v_mems,
        _mems_p, _durations_p, _vertexes_p, _v_mems_p,
        timeperiod, extremes_f32, to_calc_length
    )


def mark_neutral_memory(df: DataFrame, wsconfig: dict, to_calc_length: int,
                        attrs: list[str], n_attrs: list[str], timeperiod: int, params: list[np.float64]):
    num = len(attrs)
    if num == 0:
        return

    n = len(df)
    # 1. 预先准备好所有列，确保内存占坑（指针同步的前提）
    # 注意：mem 和 mem-prev 必须拥有独立的内存空间
    suffix_list = ['-mem', '-duration', '-mem-prev', '-duration-prev']
    for n_attr in n_attrs:
        for suffix in suffix_list:
            col = n_attr + suffix
            if col not in df.columns:
                df[col] = np.float32(0.0)  # 明确使用 float32 占坑

    # 2. 准备视图列表（List of Views）
    # 这样传给 Numba 的是直接指向 df 内部的指针
    _attrs_views = [df[a].values.astype('float32') if df[a].dtype != 'float32' else df[a].values for a in attrs]

    _mems = [df[f'{na}-mem'].values for na in n_attrs]
    _durs = [df[f'{na}-duration'].values for na in n_attrs]
    _mems_p = [df[f'{na}-mem-prev'].values for na in n_attrs]
    _durs_p = [df[f'{na}-duration-prev'].values for na in n_attrs]

    # 3. 处理全量计算时的重置逻辑（使用 [:] 原地重置）
    if to_calc_length >= n:
        for i in range(num):
            _mems[i][:] = 0.0
            _durs[i][:] = 0.0
            _mems_p[i][:] = 0.0
            _durs_p[i][:] = 0.0

    # 4. 参数转换
    params_f32 = np.array(params, dtype='float32')

    # 5. 调用核心计算（原地修改视图）
    # 确保你的底层 _mark_neutral_memory 接受的是这些 list of arrays
    _mark_neutral_memory(
        _attrs_views, _mems, _durs, _mems_p, _durs_p,
        timeperiod, params_f32, to_calc_length
    )

@nb.njit
def _mark_neutral_memory(attrs: [], mems: [], durations: [], mems_prev: [], durations_prev: [], timeperoid: int, params: list[np.float32],
                         to_calc_length: int):
    n = len(attrs[0])
    num = len(attrs)
    n_start = max(n - to_calc_length, 0)
    m_cut = np.exp((-3))
    m_0 = np.exp(-1 / timeperoid)
    for j in range(0, num):
        attr = attrs[j]
        mem = mems[j]
        mem_prev = mems_prev[j]
        duration = durations[j]
        duration_prev = durations_prev[j]
        param = params[j]
        for i in range(n_start, n):
            if i == 0:
                mem[i] = 0
                duration[i] = 0
                mem_prev[i] = 0
                duration_prev[i] = 0
            switch = False
            if np.abs(attr[i]) < param:
                _sign = np.sign(attr[i])
                if np.abs(mem[i - 1]) < 0.9 or np.sign(mem[i - 1]) != _sign:
                    switch = True
                mem[i] = _sign
                if switch:
                    duration[i] = _sign
                else:
                    duration[i] = duration[i - 1] + _sign
            elif mem[i - 1] != 0:
                mem[i] = mem[i - 1] * m_0
                duration[i] = duration[i - 1]
                if np.abs(mem[i]) < m_cut:
                    mem[i] = 0
            if switch:
                mem_prev[i] = mem[i - 1] * m_0
                duration_prev[i] = duration[i - 1]
                if np.abs(mem_prev[i]) < m_cut:
                    mem_prev[i] = 0
            elif mem_prev[i - 1] != 0:
                mem_prev[i] = mem_prev[i - 1] * m_0
                duration_prev[i] = duration_prev[i - 1]
                if np.abs(mem_prev[i]) < m_cut:
                    mem_prev[i] = 0


def mark_extreme_integral(df: pd.DataFrame, wsconfig: dict, to_calc_length: int,
                          attrs: list[str], n_attrs: list[str], sma_length: int, params: list[np.float64]):
    num = len(attrs)
    if num == 0:
        return
    n = len(df)

    # 1. 独立初始化列，确保内存物理隔离
    for na in n_attrs:
        for suffix in ['-integral', '-amount']:
            col = na + suffix
            if col not in df.columns:
                df[col] = np.float32(0.0)

    # 2. 准备视图列表
    # 强制统一为 float32 以匹配底层内核，避免 Numba 内部转换开销
    def get_f32_view(name):
        if df[name].dtype != np.float32:
            df[name] = df[name].astype(np.float32)
        return df[name].values

    _attrs = [get_f32_view(a) for a in attrs]
    _integs = [get_f32_view(f'{na}-integral') for na in n_attrs]
    _amounts = [get_f32_view(f'{na}-amount') for na in n_attrs]

    # 3. 优化 SMA 计算：使用底层工具减少对象创建
    # 注意：ta.SMA 返回的是 float64，我们将其转为 float32 视图
    _means = []
    for a in attrs:
        # 这里的 SMA 依然是全量计算，如果性能遇到瓶颈，建议改写为 Numba 版本的增量 SMA
        s_mean = ta.SMA(df[a].values.astype('float64'), timeperiod=sma_length)
        _means.append(s_mean.astype('float32'))

    # 4. 全量计算时的内存清理 [:]
    if to_calc_length >= n:
        for i in range(num):
            _integs[i][:] = 0.0
            _amounts[i][:] = 0.0

    # 5. 参数对齐
    params_f32 = np.array(params, dtype='float32')

    # 6. 调用内核（原地修改 _integs 和 _amounts）
    _mark_extreme_integral(
        _attrs, _means, _integs, _amounts,
        sma_length, params_f32, to_calc_length
    )


@nb.njit
def _mark_extreme_integral(attrs: [], means: [], integs: [], amounts: [], sma_length: int, params: list[np.float64],
                           to_calc_length: int):
    n = len(attrs[0])
    num = len(attrs)
    n_start = max(n - to_calc_length, sma_length)
    for j in range(0, num):
        attr = attrs[j]
        mean = means[j]
        integ = integs[j]
        amount = amounts[j]
        param = params[j]
        for i in range(n_start, n):
            if i == 0:
                last_integ = 0
                last_amount = 0
            else:
                last_integ = integ[i - 1]
                last_amount = amount[i - 1]
            dif = np.abs(attr[i]) - np.abs(mean[i])
            if np.abs(attr[i]) > param or (dif < 0 and integ[i] != 0):
                _sign = np.sign(attr[i])
                if np.sign(last_integ) != _sign:
                    last_integ = 0
                    last_amount = 0
                if np.abs(last_integ) + dif < 0:
                    integ[i] = 0
                    amount[i] = 0
                else:
                    integ[i] = (_sign * dif) + last_integ
                    amount[i] = last_amount + 1
            else:
                integ[i] = last_integ
                amount[i] = last_amount


@nb.njit
def _mark_extreme_integral_old(attrs: [], integs: [], amounts: [], timeperoid: int, params: list[np.float64],
                               to_calc_length: int):
    n = len(attrs[0])
    num = len(attrs)
    n_start = max(n - to_calc_length, 0)
    for j in range(0, num):
        attr = attrs[j]
        integ = integs[j]
        amount = amounts[j]
        param = params[j]
        for i in range(n_start, n):
            if i == 0:
                last_integ = 0
                last_amount = 0
            else:
                last_integ = integ[i - 1]
                last_amount = amount[i - 1]
            dif = np.abs(attr[i]) - param
            if dif > 0:
                _sign = np.sign(attr[i])
                integ[i] = _sign * dif + last_integ
                amount[i] = _sign + last_amount
            else:
                integ[i] = last_integ
                amount[i] = 0
            if i > timeperoid:
                difx = np.abs(attr[i - timeperoid]) - param
                if difx > 0:
                    _signx = 0 - np.sign(attr[i - timeperoid])
                    integ[i] = _signx * difx + integ[i]


@nb.njit
def _calculate_ratr(_attr, _n_attr, timeperiod, to_calc_length):
    n = len(_attr)
    first_valid_idx = 0
    for k in range(n):
        if not np.isnan(_attr[k]):
            first_valid_idx = k
            break
    if first_valid_idx >= n - 1:
        return
    start = max(n - to_calc_length, first_valid_idx + timeperiod + 1)
    if n - to_calc_length < first_valid_idx + timeperiod + 1:
        _sum = 0
        for i in range(first_valid_idx + 1, first_valid_idx + timeperiod + 1):
            _sum = _sum + np.abs(_attr[i] - _attr[i - 1])
        _n_attr[first_valid_idx + timeperiod] = _sum / timeperiod
    for i in range(start, n):
        tr = np.abs(_attr[i] - _attr[i - 1])
        _n_attr[i] = _n_attr[i - 1] + (tr - _n_attr[i - 1]) / timeperiod


# 根据每时间周期平均close变化计算的等效ATR
def calculate_ratr(df: DataFrame, wsconfig: dict, to_calc_length: int,
                   attr: str, n_attr: str, timeperiod: int):
    n = len(df)
    col_name = f'{n_attr}'

    # 1. 防御性初始化：确保列已存在并统一类型（float32 通常更适合量化特征库）
    if col_name not in df.columns:
        df[col_name] = np.zeros(n)

    # 3. 获取物理内存视图 (View)
    _attr = df[attr].values
    _n_attr = df[col_name].values

    # 4. 全量计算时的重置逻辑（使用 [:] 原地抹除内容，不改变内存地址）
    if to_calc_length >= n:
        _n_attr[:] = 0.0

    # 5. 调用内核（原地修改 _n_attr）
    _calculate_ratr(_attr, _n_attr, timeperiod, to_calc_length)


def calculate_hurst_old(df: DataFrame, wsconfig: dict, to_calc_length: int, attr: str, timeperoid: int) -> pd.Series:
    """
        主要滚动计算函数，将数据准备好后，交由 Numba 优化函数处理。
    """
    ver = f"0002_timeperoid_{timeperoid}_optimized"
    load = _auto_load(wsconfig, "calc_hurst_rs_optimized", attr, ver)
    if load is not None:
        return load

    s = df[attr]
    # 1. 预先计算整个序列的对数收益率 (Log Returns)
    # 避免在循环中重复计算 np.log 和 np.diff
    log_prices = np.log(s.values)
    log_returns_full = np.diff(log_prices)

    # 2. 调用 Numba 优化函数进行滚动计算
    # 将整个 log_returns 数组和滚动窗口大小传递给 Numba
    # Numba 函数将在 C 速度下处理滚动循环
    hurst_rs_array = _calculate_rolling_hurst_numba(log_returns_full, timeperoid)
    final_result = np.full(len(s), np.nan)

    target_slice = final_result[timeperoid:]
    print(f'target={len(target_slice)}, hurst={len(hurst_rs_array)}')
    if len(target_slice) != len(hurst_rs_array):
        # 临时调试：如果遇到这个错误，请返回这里，检查 L_log, L, timeperoid 之间的关系
        # 例如：print(f"Target len: {len(target_slice)}, Source len: {len(source_array)}")
        # 假设我们在 log_returns 的最后一个点结束计算，则结果应该比 log_returns 短 timeperoid - 1。

        # 简单粗暴但有效的方法：直接用 Numba 结果的长度来确定切片
        target_slice_start = len(s) - len(hurst_rs_array)
        final_result[target_slice_start:] = hurst_rs_array
    else:
        # 如果长度匹配，使用原始逻辑 (但这次应该不会再错)
        final_result[timeperoid:] = hurst_rs_array

    _auto_save(wsconfig, final_result, "calc_hurst_rs_optimized", s.name, ver)
    return final_result


def calculate_hurst(df: DataFrame, wsconfig: dict, to_calc_length: int, attr: str, timeperoid: int, timeframe_size: int=1, target_attr: str = 'hurst') -> np.ndarray:
    ver = f"0004_timeperoid_{timeperoid}_timeframesize_{timeframe_size}"

    # 1. 无论如何，先在 df 内部“占坑”
    # 这一步保证了 target_attr 在 df 内部拥有一个稳定的物理地址
    if target_attr not in df.columns:
        df[target_attr] = np.full(len(df), np.nan, dtype=np.float64)

    # 获取稳定的内存视图
    target_view = df[target_attr].values

    load = _auto_load(wsconfig, "calculate_hurst", attr, ver)
    if load is not None:
        # 【关键】：使用 [:] 原位灌入，不改变 target_attr 的引用地址
        fill_len = min(len(load), len(target_view))
        target_view[-fill_len:] = load[-fill_len:]
        return target_view

    s = df[attr].values

    # 实际传入 Numba 内部参与 R/S 拟合的点数
    actual_window_size = int(timeperoid / timeframe_size)
    # 确保采样长度是步长的整数倍，避免切片越界
    effective_raw_window = actual_window_size * timeframe_size
    h_res = calc_hurst(s, actual_window_size, timeframe_size)

    # 4. 【关键】：直接原地修改 target_view
    # 这样就不需要再在外部执行 df['hurst'] = ...
    target_view[:] = np.nan  # 先清空旧值
    start_loc = (timeperoid - effective_raw_window)
    # 长度保护：确保写入不越界
    write_len = min(len(h_res), len(target_view) - start_loc)
    target_view[start_loc: start_loc + write_len] = h_res[:write_len]

    # 6. 保存
    _auto_save(wsconfig, target_view, "calculate_hurst", attr, ver)
    return target_view

def calc_hurst(close, window, stride):
    log_returns_full = np.concatenate([
        np.array([0.0], dtype=np.float64),
        np.diff(np.log(close))
    ])
    h_res = _calculate_rolling_dfa_hurst_numba(log_returns_full, window_size=window, stride=stride)
    head_nan_len = len(close) - len(h_res)
    h_res = np.concatenate([
        np.full(head_nan_len, np.nan, dtype=np.float64),
        h_res
    ])
    mask = ~np.isnan(h_res)
    if np.any(mask):
        first_valid_idx = np.argmax(mask)
        print(f'calc_hurst: len={len(close)}, window={window}, first_non_nan_at={first_valid_idx}')
    else:
        print(f'calc_hurst: len={len(close)}, window={window}, ALL values are NaN!')
    # ------------------
    return h_res

@njit  # 强烈建议配合 Numba
def _calculate_rs_values_for_window(log_returns_segment):
    n = len(log_returns_segment)
    max_k = n // 2
    segment_min = 20

    # 1. 预先计算最大可能的循环次数
    # k 从 segment_min 变到 max_k，最大可能产生的个数是：
    max_possible_size = max(0, max_k - segment_min + 1)

    # 如果长度不足以计算，直接返回空数组
    if max_possible_size == 0:
        return np.empty(0), np.empty(0)

    # 2. 预分配内存区
    # 使用 np.empty 会比 np.zeros 快一点，因为不初始化内存
    n_values_pool = np.empty(max_possible_size, dtype=np.float64)
    rs_values_pool = np.empty(max_possible_size, dtype=np.float64)

    # 记录实际写入了多少个点
    write_idx = 0

    for k in range(segment_min, max_k + 1):
        num_segments = n // k
        if num_segments < 3:
            continue

        rs_per_segment_sum = 0.0
        rs_per_segment_count = 0

        for i in range(num_segments):
            start_idx = i * k
            end_idx = start_idx + k
            segment = log_returns_segment[start_idx:end_idx]

            # 内部计算逻辑（手动展开 mean/std 性能会更好，Numba 优化得很好）
            mean_segment = np.mean(segment)
            # 这里的 cumsum 如果在 Numba 下会自动优化，不需要额外 append
            cumulative_deviation = np.cumsum(segment - mean_segment)

            R = np.max(cumulative_deviation) - np.min(cumulative_deviation)
            S = np.std(segment)

            if S > 1e-9:
                rs_per_segment_sum += R / S
                rs_per_segment_count += 1

        if rs_per_segment_count > 0:
            avg_rs = rs_per_segment_sum / rs_per_segment_count
            # 3. 直接写入预分配的内存区
            n_values_pool[write_idx] = float(k)
            rs_values_pool[write_idx] = avg_rs
            write_idx += 1

    # 4. 只返回有效的部分
    return n_values_pool[:write_idx], rs_values_pool[:write_idx]


@nb.njit(parallel=True)
def _calculate_rolling_hurst_numba(log_returns_full, window_size, stride=1):
    """
    在 Numba 内部执行高效的滚动计算和 polyfit。
    """
    full_length = len(log_returns_full)

    if full_length < window_size:
        return np.empty(0)

    # 结果数组的长度
    result_length = full_length - window_size
    hurst_rs = np.zeros(result_length)

    # Python 外部的 polyfit 必须在这里实现
    # 由于 np.polyfit 不支持 nopython=True，我们需要手动实现线性回归
    for i in nb.prange(result_length):
        start = i
        end = start + window_size
        segment = log_returns_full[start:end:stride]

        # 调用核心 R/S 计算
        n_values, rs_values = _calculate_rs_values_for_window(segment)

        # 在 Numba 中手动实现线性回归 (代替 np.polyfit)
        if len(n_values) >= 5:
            log_n = np.log(n_values)
            log_rs = np.log(rs_values)

            # 手动计算线性回归 (最小二乘法)
            n = len(log_n)
            sum_x = np.sum(log_n)
            sum_y = np.sum(log_rs)
            sum_xy = np.sum(log_n * log_rs)
            sum_xx = np.sum(log_n * log_n)

            # 斜率 (Slope) 公式
            numerator = n * sum_xy - sum_x * sum_y
            denominator = n * sum_xx - sum_x * sum_x

            if denominator != 0:
                hurst_exponent = numerator / denominator
                if hurst_exponent < 0.0:
                    hurst_rs[i] = 0.0
                elif hurst_exponent > 1.0:
                    hurst_rs[i] = 1.0
                else:
                    hurst_rs[i] = hurst_exponent
            else:
                hurst_rs[i] = np.nan
        else:
            hurst_rs[i] = np.nan

    return hurst_rs

@nb.njit(parallel=True)
def _calculate_rolling_dfa_hurst_numba(log_returns_full, window_size, stride=1):
    full_length = len(log_returns_full)
    if full_length < window_size:
        return np.empty(0)

    # 结果数组的长度：依然对应 dfdst 的每一行
    result_length = full_length - window_size
    alpha_res = np.zeros(result_length)

    # 采样后的有效窗口长度
    effective_n = window_size // stride

    max_s = effective_n // 4
    min_s = 10

    temp_scales = []
    curr_s = float(min_s)
    # 使用 1.5 倍增长，可以在 [10, 36] 之间拿到 10, 15, 22, 33 约4个点
    # 如果窗口大，点会更多
    while curr_s <= max_s:
        temp_scales.append(int(curr_s))
        curr_s *= 1.5

    active_scales = np.array(temp_scales)
    num_active = len(active_scales)

    # 保护：如果回归点数太少（少于3个），DFA 失去意义
    if num_active < 3:
        return np.full(result_length, np.nan)
    for i in nb.prange(result_length):
        start = i
        end = start + window_size

        # --- 关键修改：按 stride 采样 ---
        # 这里的 segment 长度约为 window_size // stride
        segment = log_returns_full[start:end:stride]

        # 1. 集成序列 (在采样后的维度上进行)
        walk = np.cumsum(segment - np.mean(segment))
        current_n = len(walk)

        f_n = np.zeros(len(active_scales))

        for s_idx in range(len(active_scales)):
            n = active_scales[s_idx]
            num_segments = current_n // n
            if num_segments < 2: continue

            rms_sum_sq = 0.0
            for j in range(num_segments):
                w_sub = walk[j * n: (j + 1) * n]

                # 2. 线性去趋势
                x = np.arange(float(n))
                y = w_sub

                sum_x = (n * (n - 1)) / 2.0
                sum_xx = (n * (n - 1) * (2 * n - 1)) / 6.0
                sum_y = np.sum(y)
                sum_xy = np.sum(x * y)

                denom = n * sum_xx - sum_x ** 2
                if abs(denom) > 1e-12:
                    slope = (n * sum_xy - sum_x * sum_y) / denom
                    intercept = (sum_y - slope * sum_x) / n

                    # 3. 计算残差
                    res_sq_sum = 0.0
                    for k_idx in range(n):
                        res_sq_sum += (y[k_idx] - (slope * x[k_idx] + intercept)) ** 2
                    rms_sum_sq += res_sq_sum / n

            f_n[s_idx] = np.sqrt(rms_sum_sq / num_segments)

        # 4. 线性回归求 alpha
        valid_mask = f_n > 1e-12
        if np.sum(valid_mask) >= 3:
            log_scales = np.log(active_scales[valid_mask].astype(np.float64))
            log_fn = np.log(f_n[valid_mask])

            # 手动线性回归
            n_reg = len(log_scales)
            sx, sy, sxy, sxx = np.sum(log_scales), np.sum(log_fn), np.sum(log_scales * log_fn), np.sum(log_scales ** 2)
            d = n_reg * sxx - sx ** 2
            if abs(d) > 1e-12:
                alpha_res[i] = (n_reg * sxy - sx * sy) / d
            else:
                alpha_res[i] = np.nan
        else:
            alpha_res[i] = np.nan

    return alpha_res

@nb.njit
def _mark_next_ups(values: np.ndarray, gaps: np.ndarray, rs: np.ndarray, timemax: int):
    n = len(values)
    for i in range(0, n):
        _end = min(n, i + timemax)
        _v = values[i] + gaps[i]
        for j in range(i, _end):
            if values[j] >= _v:
                rs[i] = j - i
                break


@nb.njit
def _mark_next_downs(values: np.ndarray, gaps: np.ndarray, rs: np.ndarray, timemax: int):
    n = len(values)
    for i in range(0, n):
        _end = min(n, i + timemax)
        _v = values[i] - gaps[i]
        for j in range(i, _end):
            if values[j] <= _v:
                rs[i] = j - i
                break


def mark_next_ups(df: DataFrame, wsconfig: dict, to_calc_length: int, timemax: int, param: np.float64,
                  attr: str = None) -> np.ndarray:
    _display_attr = 'None' if attr is None else attr
    ver = f"0003_timemax_{timemax}_param_{param}_optimized"
    # load = _auto_load(wsconfig, "mark_next_ups", _display_attr, ver)
    # if load is not None:
    #    return load
    if attr is None:
        gaps = np.full(len(df), param, dtype=np.float64)
    else:
        gaps = df[attr].values * param
    ups = np.full(len(df), timemax, dtype=np.float64)
    _mark_next_ups(df['close'].values, gaps, ups, timemax)
    # _auto_save(wsconfig, ups, "mark_next_ups", _display_attr, ver)
    return ups


def mark_next_downs(df: DataFrame, wsconfig: dict, to_calc_length: int, timemax: int, param: np.float64,
                    attr: str = None) -> np.ndarray:
    _display_attr = 'None' if attr is None else attr
    ver = f"0003_timemax_{timemax}_param_{param}_optimized"
    # load = _auto_load(wsconfig, "mark_next_downs", _display_attr, ver)
    # if load is not None:
    #    return load
    if attr is None:
        gaps = np.full(len(df), param, dtype=np.float64)
    else:
        gaps = df[attr].values * param
    dps = np.full(len(df), timemax, dtype=np.float64)
    _mark_next_downs(df['close'].values, gaps, dps, timemax)
    # _auto_save(wsconfig, ups, "mark_next_downs", _display_attr, ver)
    return dps


@nb.njit
def _calculate_circle_length_ema(sides: np.ndarray, emas: np.ndarray, init_avg: np.float64, alpha=0.01):
    n = len(sides)
    last_trigger_idx = -1
    avg = init_avg
    emas[0] = avg
    for i in range(1, n):
        # 探测 -1 -> 1 的翻转瞬间
        if sides[i - 1] == -1 and sides[i] == 1:
            if last_trigger_idx == -1:
                # 第一次发现翻转，无法计算距离，仅记录索引
                last_trigger_idx = i
            else:
                # 计算两个翻转点之间的行数距离
                cycle_len = i - last_trigger_idx
                avg = avg + (cycle_len - avg) * alpha
                last_trigger_idx = i
        emas[i] = avg


def calculate_circle_length_ema(df: DataFrame, wsconfig: dict, to_calc_length: int,
                                side_attr: str, ema_attr: str,
                                init_avg: np.float64, alpha=0.01) -> DataFrame:
    n = len(df)
    if to_calc_length >= n:
        df[ema_attr] = np.full(n, 0, dtype=np.float64)
    _calculate_circle_length_ema(df[side_attr].values, df[ema_attr].values, init_avg, alpha=alpha)
    return df

@njit
def calc_symbolic_entropy(close: np.ndarray, to_calc_length: int, timeperoid=720, pattern_len=3, combine_period=1):
    """
    close: 1m 收盘价序列
    lookback: 统计分布的窗口大小 (以1m为单位，如720代表最近12小时)
    pattern_len: 模式长度 L (如3, 4, 5)
    combine_period: 合并周期 C (如1代表1m, 5代表5m)
    """
    n = len(close)
    entropy_vec = np.zeros(n)

    # 模式数量 2^L
    num_patterns = 1 << pattern_len
    # 计数器
    counts = np.zeros(num_patterns, dtype=np.int32)

    # 1. 预计算符号 (当前值相对于 C 周期前的值)
    symbols = np.zeros(n, dtype=np.int32)
    for i in range(combine_period, n):
        if close[i] >= close[i - combine_period]:
            symbols[i] = 1
        else:
            symbols[i] = 0

    # 2. 预计算每个位置作为结尾的模式索引 (Pattern Index)
    # 例如 C=5, L=3, 则模式由 symbols[i], symbols[i-5], symbols[i-10] 组成
    pattern_indices = np.zeros(n, dtype=np.int32)
    offset = (pattern_len - 1) * combine_period
    for i in range(offset, n):
        idx = 0
        for k in range(pattern_len):
            # 位运算合成索引
            bit = symbols[i - k * combine_period]
            idx = (idx << 1) | bit
        pattern_indices[i] = idx

    # 3. 滑动窗口统计 (O(1) 递增更新)
    # 有效起始点：需满足模式长度和统计窗口
    start_idx = timeperoid + offset
    if n <= start_idx:
        return entropy_vec

    # 初始窗口填充
    for j in range(start_idx - timeperoid + 1, start_idx + 1):
        counts[pattern_indices[j]] += 1

    for i in range(start_idx, n):
        if i > start_idx:
            # 移入新模式，移出旧模式
            new_p = pattern_indices[i]
            old_p = pattern_indices[i - timeperoid]
            counts[new_p] += 1
            counts[old_p] -= 1

        # 计算香农熵
        e = 0.0
        total = timeperoid
        for c in counts:
            if c > 0:
                p = c / total
                e -= p * np.log2(p)
        entropy_vec[i] = e

    return entropy_vec


@njit
def calc_fractal_er(close, to_calc_length: int, timeperoid=720, combine_period=1):
    """
    lookback: 统计窗口 (以1m为单位)
    combine_period: 合并周期 C (如5代表观察5m级别的路径效率)
    """
    n = len(close)
    er_vec = np.zeros(n)

    # 路径总和 (Rolling Sum)
    path_length = 0.0

    # 预计算每一步的“步长” (基于 C 周期)
    # 注意：为了量纲一致，我们依然在 1m 步长上滑动，但比较的是 C 周期的变动
    steps = np.zeros(n)
    for i in range(combine_period, n):
        steps[i] = abs(close[i] - close[i - combine_period])

    for i in range(combine_period, n):
        path_length += steps[i]
        if i >= timeperoid:
            path_length -= steps[i - timeperoid]

            if i >= timeperoid:
                # 净位移
                disp = close[i] - close[i - timeperoid]
                if path_length > 0:
                    er_vec[i] = disp / path_length
                else:
                    er_vec[i] = 0.0

    return er_vec


def calculate_symbolic_entropy(df: DataFrame, wsconfig: dict, to_calc_length: int, timeperoid: int, combine_period=1, pattern_len=3):
    return np.log(3.001 - calc_symbolic_entropy(df['close'].values, to_calc_length, timeperoid, pattern_len, combine_period))


def calulate_freactal_er(df: DataFrame, wsconfig: dict, to_calc_length: int, timeperoid: int, combine_period=1):
    return np.arctanh(calc_fractal_er(df['close'].values, to_calc_length, timeperoid, combine_period) * 0.9999)


@nb.jit(nopython=True)
def _calculate_bjs(close: np.ndarray, step: np.ndarray,
                   tm_length: np.ndarray, bjsshort: np.ndarray, bjslong: np.ndarray,
                   min_ratio: float, max_ratio: float, to_calc_length: int):
    n = len(close)
    # print(close)
    max_len = int(np.rint(tm_length[-1] * max_ratio))
    start = max(n - to_calc_length, max_len)
    # _sqrts = np.zeros(max_len - min_len, dtype=np.float64)
    # for j in range(min_len, max_len):
    #    _sqrts[j - min_len] = np.sqrt(j / min_len)
    for i in range(start, n):
        if np.isnan(step[i]) or step[i] == 0:
            continue
        min_len = int(np.rint(tm_length[i] * min_ratio))
        max_len = int(np.rint(tm_length[i] * max_ratio))
        bjsshort[i] = (close[i] - close[i - min_len]) / step[i]
        bjslong[i] = 0
        for j in range(min_len, max_len):
            d = close[i] - close[i - j]
            # if np.abs(d) < step[i]:
            #    continue
            r = d / step[i] / (np.sqrt(j / (min_len + 1e-9)) + 1e-9)  # _sqrts[j - min_len]
            if np.abs(r) > np.abs(bjslong[i]):
                bjslong[i] = r


def calculate_bjs(df: DataFrame, wsconfig: dict, to_calc_length: int, prefix: str, val: int, min_ratio: float, max_ratio: float):
    n = len(df)
    close = df['close'].values
    step = df[f'diff{prefix}-{val}-mean-long'].values
    tm_length = df[f'tm-mean-{prefix}'].values
    if to_calc_length >= n:
        bjsshort = np.zeros(n)
        bjslong = np.zeros(n)
    else:
        bjsshort = df[f'bjsshort{prefix}-{val}'].values
        # bjslong = df[f'bjslong_o{prefix}-{val}'].values
        bjslong = df[f'bjslong{prefix}-{val}'].values
    _calculate_bjs(close, step, tm_length, bjsshort, bjslong, min_ratio, max_ratio, to_calc_length)
    df[f'bjsshort{prefix}-{val}'] = bjsshort
    df[f'bjslong{prefix}-{val}'] = bjslong
    # df[f'bjslong{prefix}-{val}'] = ta.EMA(df[f'bjslong_o{prefix}-{val}'], timeperiod=7)

def talib_ema(data, period):
    return _ema_kernel(data, period)