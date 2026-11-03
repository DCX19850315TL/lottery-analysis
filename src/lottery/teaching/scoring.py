"""双色球与大乐透共用的 z 分数、softmax 与带种子选池。"""

import numpy as np
import pandas as pd


def zscore(series: pd.Series) -> pd.Series:
    """返回 z 分数；标准差为 0 时返回全 0。"""
    std = series.std(ddof=0)
    if std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def softmax(weights: pd.Series) -> np.ndarray:
    """把权重转成和为 1 的概率向量；全零权重得到均匀分布。"""
    values = np.asarray(weights, dtype=float)
    exp = np.exp(values - values.max())
    return exp / exp.sum()


def select_pool(scores: pd.Series, size: int, seed: int) -> list[int]:
    """按 seed 打乱同分次序后按分数降序取前 size 个，消除小号偏置。"""
    if not 1 <= size <= len(scores):
        raise ValueError(f"选池大小必须在 1..{len(scores)}，实际 {size}")
    items = list(scores.items())
    np.random.default_rng(seed).shuffle(items)
    ordered = sorted(items, key=lambda item: -item[1])
    return [int(number) for number, _ in ordered[:size]]
