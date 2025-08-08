import logging
from datetime import datetime, time
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from pandas import DataFrame, Timestamp

from freqtrade.configuration import sanitize_config
from freqtrade.constants import LAST_BT_RESULT_FN
from freqtrade.enums.runmode import RunMode
from freqtrade.ft_types import BacktestResultType
from freqtrade.misc import dump_json_to_file, file_dump_json
from freqtrade.optimize.backtest_caching import get_backtest_metadata_filename

logger = logging.getLogger(__name__)


def file_dump_joblib(file_obj: BytesIO, data: Any, log: bool = True) -> None:
    """
    Dump object data into a file
    :param filename: file to create
    :param data: Object data to save
    :return:
    """
    import joblib

    joblib.dump(data, file_obj)


def _generate_filename(recordfilename: Path, appendix: str, suffix: str) -> Path:
    """
    Generates a filename based on the provided parameters.
    :param recordfilename: Path object, which can either be a filename or a directory.
    :param appendix: use for the filename. e.g. backtest-result-<datetime>
    :param suffix: Suffix to use for the file, e.g. .json, .pkl
    :return: Generated filename as a Path object
    """
    if recordfilename.is_dir():
        filename = (recordfilename / f"backtest-result-{appendix}").with_suffix(suffix)
    else:
        filename = Path.joinpath(
            recordfilename.parent, f"{recordfilename.stem}-{appendix}"
        ).with_suffix(suffix)
    return filename


def store_backtest_extra_data(
        config: dict,
        data: pd.DataFrame,
        index: int,
        metas: dict
) -> Path:
    recordfilename: Path = config["exportfilename"]
    base_filename = _generate_filename(recordfilename, metas['dt_appendix'], "." + metas['strat_name'])
    strategy_zip_filename = _generate_filename(recordfilename, metas['dt_appendix'], "." + metas['strat_name'] + "." +
                                               metas['pair'].replace("/", "-")
                                               + ".extra." + index + ".zip")
    # Create zip file and add the files
    with ZipFile(strategy_zip_filename, "w", ZIP_DEFLATED) as zipf:
        entire_data_name = f"{base_filename.stem}_extra_data." + index + ".feather"
        entire_data_buf = BytesIO()
        data.reset_index().to_feather(
            entire_data_buf, compression_level=9, compression="lz4"
        )
        entire_data_buf.seek(0)
        zipf.writestr(entire_data_name, entire_data_buf.getvalue())
    return strategy_zip_filename


def _generate_store_filename(fname: Path, func: str, version: str, pair: str, timeframe: str,
                             start_dt: Timestamp, size: int) -> Path:
    path = fname
    pair = pair.replace("/", "-").replace(":", "_")
    path = path.joinpath("backtest-vars") \
        .joinpath(start_dt.strftime("%Y-%m-%d_%H-%M-%S") + "_" + timeframe + "_" + str(size))
    path.mkdir(parents=True, exist_ok=True)
    path = path.joinpath(pair + "_" + func + "_" + version).with_suffix(".zip")
    return path


def store_backtest_series_data(
        config: dict,
        data: pd.Series,
        func: str,
        version: str
) -> Path:
    pathname = _generate_store_filename(config["user_data_dir"], func, version,
                                        config['pair'], config['timeframe'], config['date'], config['size'])
    with ZipFile(pathname, "w", ZIP_DEFLATED) as zipf:
        entire_data_name = f"{func}_{version}.feather"
        entire_data_buf = BytesIO()
        data.copy().rename('value').reset_index().to_feather(
            entire_data_buf, compression_level=9, compression="lz4"
        )
        entire_data_buf.seek(0)
        zipf.writestr(entire_data_name, entire_data_buf.getvalue())
    return pathname


def load_backtest_series_data(
        config: dict,
        func: str,
        version: str
) -> np.array:
    pathname = _generate_store_filename(config["user_data_dir"], func, version,
                                        config['pair'], config['timeframe'], config['date'], config['size'])

    try:
        with ZipFile(pathname, "r") as zipf:
            # 内部文件名是这样生成的：f"{value}_{version}.feather"
            entire_data_name = f"{func}_{version}.feather"

            # 检查文件是否存在于压缩包中
            if entire_data_name not in zipf.namelist():
                raise FileNotFoundError(f"在压缩文件 {pathname} 中未找到文件: {entire_data_name}")

            with zipf.open(entire_data_name, "r") as feather_file:
                entire_data_buf = BytesIO(feather_file.read())

                # 读取 Feather 文件，它将是一个 DataFrame，因为你存储时调用了 .reset_index()
                df = pd.read_feather(entire_data_buf)

                # 将 DataFrame 转换回 Series
                # 由于存储时调用了 reset_index()，原始的 Series 索引会变成 DataFrame 的第一列（通常是 'index'）
                # 并且 Series 的值会存储在另一个列中，这个列的名字就是 'value' 参数的值。
                # 所以我们需要将 'index' 列设置为索引，并选择 'value' 列作为 Series。

                # 假设原始索引列名为 'index' (reset_index() 的默认行为)
                if 'index' not in df.columns or 'value' not in df.columns:
                    return None
    except FileNotFoundError:
        return None
    return df['value'].values


def store_backtest_entire_data(
        config: dict,
        data: pd.DataFrame,
        metas: dict
) -> Path:
    recordfilename: Path = config["exportfilename"]
    base_filename = _generate_filename(recordfilename, metas['dt_appendix'], "." + metas['strat_name'])
    strategy_zip_filename = _generate_filename(recordfilename, metas['dt_appendix'], "." + metas['strat_name'] + "."
                                               + metas['pair'].replace("/", "-").replace(":", "_") + ".zip")

    # Create zip file and add the files
    with ZipFile(strategy_zip_filename, "w", ZIP_DEFLATED) as zipf:
        entire_data_name = f"{base_filename.stem}_entire_data.feather"
        entire_data_buf = BytesIO()
        data.reset_index().to_feather(
            entire_data_buf, compression_level=9, compression="lz4"
        )
        entire_data_buf.seek(0)
        zipf.writestr(entire_data_name, entire_data_buf.getvalue())
    return strategy_zip_filename


def store_backtest_results(
    config: dict,
    stats: BacktestResultType,
    dtappendix: str,
    *,
    market_change_data: DataFrame | None = None,
    analysis_results: dict[str, dict[str, DataFrame]] | None = None,
    strategy_files: dict[str, str] | None = None,
) -> Path:
    """
    Stores backtest results and analysis data in a zip file, with metadata stored separately
    for convenience.
    :param config: Configuration dictionary
    :param stats: Dataframe containing the backtesting statistics
    :param dtappendix: Datetime to use for the filename
    :param market_change_data: Dataframe containing market change data
    :param analysis_results: Dictionary containing analysis results
    """
    recordfilename: Path = config["exportfilename"]
    zip_filename = _generate_filename(recordfilename, dtappendix, ".zip")
    base_filename = _generate_filename(recordfilename, dtappendix, "")
    json_filename = _generate_filename(recordfilename, dtappendix, ".json")

    # Store metadata separately with .json extension
    file_dump_json(get_backtest_metadata_filename(json_filename), stats["metadata"])

    # Store latest backtest info separately
    latest_filename = Path.joinpath(zip_filename.parent, LAST_BT_RESULT_FN)
    file_dump_json(latest_filename, {"latest_backtest": str(zip_filename.name)}, log=False)

    # Create zip file and add the files
    with ZipFile(zip_filename, "w", ZIP_DEFLATED) as zipf:
        # Store stats
        stats_copy = {
            "strategy": stats["strategy"],
            "strategy_comparison": stats["strategy_comparison"],
        }
        stats_buf = StringIO()
        dump_json_to_file(stats_buf, stats_copy)
        zipf.writestr(json_filename.name, stats_buf.getvalue())

        config_buf = StringIO()
        dump_json_to_file(config_buf, sanitize_config(config["original_config"]))
        zipf.writestr(f"{base_filename.stem}_config.json", config_buf.getvalue())

        for strategy_name, strategy_file in (strategy_files or {}).items():
            # Store the strategy file and its parameters
            strategy_buf = BytesIO()
            strategy_path = Path(strategy_file)
            if not strategy_path.is_file():
                logger.warning(f"Strategy file '{strategy_path}' does not exist. Skipping.")
                continue
            with strategy_path.open("rb") as strategy_file_obj:
                strategy_buf.write(strategy_file_obj.read())
            strategy_buf.seek(0)
            zipf.writestr(f"{base_filename.stem}_{strategy_name}.py", strategy_buf.getvalue())
            strategy_params = strategy_path.with_suffix(".json")
            if strategy_params.is_file():
                strategy_params_buf = BytesIO()
                with strategy_params.open("rb") as strategy_params_obj:
                    strategy_params_buf.write(strategy_params_obj.read())
                strategy_params_buf.seek(0)
                zipf.writestr(
                    f"{base_filename.stem}_{strategy_name}.json",
                    strategy_params_buf.getvalue(),
                )

        # Add market change data if present
        if market_change_data is not None:
            market_change_name = f"{base_filename.stem}_market_change.feather"
            market_change_buf = BytesIO()
            market_change_data.reset_index().to_feather(
                market_change_buf, compression_level=9, compression="lz4"
            )
            market_change_buf.seek(0)
            zipf.writestr(market_change_name, market_change_buf.getvalue())

        # Add analysis results if present and running in backtest mode
        if (
            config.get("export", "none") == "signals"
            and analysis_results is not None
            and config.get("runmode", RunMode.OTHER) == RunMode.BACKTEST
        ):
            for name in ["signals", "rejected", "exited"]:
                if name in analysis_results:
                    analysis_name = f"{base_filename.stem}_{name}.pkl"
                    analysis_buf = BytesIO()
                    file_dump_joblib(analysis_buf, analysis_results[name])
                    analysis_buf.seek(0)
                    zipf.writestr(analysis_name, analysis_buf.getvalue())

    return zip_filename
