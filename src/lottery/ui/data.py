"""UI 数据装配：缓存历史数据、联网刷新与理论分布。"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from lottery.common.config import PROJECT_ROOT, AppConfig, get_config
from lottery.common.constants import RED_MAX, RED_MIN
from lottery.common.types import DrawRecord
from lottery.data.cleaner import CleanResult, DrawCleaner
from lottery.data.crawler import SSQCrawler
from lottery.data.repository import DrawRepository
from lottery.data.storage import DrawStorage
from lottery.theory.combinatorics import CombinatoricsEngine, FeatureName


def _build_repository(cfg: AppConfig, csv_path: Path) -> DrawRepository:
    """按配置组装存储、清洗器与采集器。"""
    return DrawRepository(DrawStorage(csv_path), DrawCleaner(), SSQCrawler(cfg.crawler))


def _csv_path() -> Path:
    """返回配置指定的历史数据 CSV 绝对路径。"""
    return PROJECT_ROOT / get_config().paths.raw_csv


@st.cache_data(show_spinner=False)
def _cached_draws(csv_path: str) -> tuple[DrawRecord, ...]:
    """缓存解析后的历史数据，避免页面每次重跑都重复解析 CSV。"""
    return tuple(_build_repository(get_config(), Path(csv_path)).all_draws())


def load_draws() -> list[DrawRecord]:
    """返回缓存的历史数据；CSV 不存在时返回空列表。"""
    return list(_cached_draws(str(_csv_path())))


def refresh_draws() -> CleanResult:
    """联网刷新历史数据并清空读取缓存，返回清洗结果。"""
    result = _build_repository(get_config(), _csv_path()).refresh()
    _cached_draws.clear()
    return result


def select_pool(pool_size: int, seed: int) -> tuple[int, ...]:
    """按种子均匀选取号码池；与数学路线 CLI 保持同一语义。"""
    rng = np.random.default_rng(seed)
    picked = rng.choice(np.arange(RED_MIN, RED_MAX + 1), size=pool_size, replace=False)
    return tuple(sorted(int(number) for number in picked))


@st.cache_data(show_spinner=False)
def sum_distribution() -> pd.Series:
    """返回全部组合的和值理论分布并缓存，首次枚举约数秒。"""
    return CombinatoricsEngine().distribution(FeatureName.SUM)
