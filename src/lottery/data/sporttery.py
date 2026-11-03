"""体育彩票历史开奖采集与存储：大乐透、排列3、排列5。"""

import os
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
import requests

from lottery.common.config import SportteryConfig
from lottery.data.crawler import CrawlError
from lottery.data.storage import StorageError

SPORTS_COLUMNS = ["issue", "draw_date", "numbers"]
_ISSUE_PATTERN = re.compile(r"\d{5}")


@dataclass(frozen=True, order=True)
class SportsDraw:
    """一期体彩开奖记录，号码保持官方顺序。"""

    issue: str
    draw_date: date
    numbers: tuple[int, ...]

    def __post_init__(self) -> None:
        """校验 5 位期号与非负号码序列。"""
        if _ISSUE_PATTERN.fullmatch(self.issue) is None:
            raise ValueError(f"体彩期号必须为 5 位数字: {self.issue!r}")
        if not self.numbers:
            raise ValueError("开奖号码不能为空")
        if any(number < 0 for number in self.numbers):
            raise ValueError("开奖号码不能为负数")


class SportteryCrawler:
    """体彩官方接口采集器；首次请求前先访问主页获取 Cookie。"""

    def __init__(self, cfg: SportteryConfig, session: requests.Session | None = None) -> None:
        """保存配置与可选会话，延迟到首次请求再预热。"""
        self._cfg = cfg
        self._session = session or requests.Session()
        self._warmed = False
        self._last_request = 0.0

    def fetch_page(self, game_no: str, page_no: int) -> tuple[list[SportsDraw], int]:
        """抓取单页，返回记录列表与官方报告的总页数。"""
        return self._parse(self._fetch(game_no, page_no))

    def fetch_all(self, game_no: str) -> list[SportsDraw]:
        """按官方页数抓取全部历史记录，最多 cfg.max_pages 页。"""
        records: list[SportsDraw] = []
        page_no, pages = 1, 1
        while page_no <= pages and page_no <= self._cfg.max_pages:
            page, pages = self.fetch_page(game_no, page_no)
            records.extend(page)
            page_no += 1
        return records

    def _fetch(self, game_no: str, page_no: int) -> dict:
        """发请求并解析 JSON；网络或 JSON 错误统一转 CrawlError。"""
        self._warm_up()
        params = {
            "gameNo": game_no,
            "provinceId": 0,
            "pageSize": self._cfg.page_size,
            "isVerify": 1,
            "pageNo": page_no,
        }
        try:
            response = self._request_with_retry(params)
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise CrawlError(f"体彩接口请求失败: gameNo={game_no} page={page_no}") from exc
        if not isinstance(payload, dict):
            raise CrawlError("体彩接口返回结构异常")
        return payload

    def _request_with_retry(self, params: dict) -> requests.Response:
        """按配置重试请求；最后一次失败仍抛原始异常。"""
        attempts = self._cfg.retry.max_attempts
        for attempt in range(1, attempts + 1):
            self._apply_rate_limit()
            try:
                response = self._session.get(
                    self._cfg.api_url,
                    params=params,
                    headers=self._headers(),
                    timeout=self._cfg.timeout_s,
                )
                response.raise_for_status()
                return response
            except requests.RequestException:
                if attempt == attempts:
                    raise
                time.sleep(self._cfg.retry.backoff_s * 2 ** (attempt - 1))
        raise AssertionError("unreachable")

    def _warm_up(self) -> None:
        """首次请求前访问主页，取得官方 WAF 所需 Cookie。"""
        if self._warmed:
            return
        self._apply_rate_limit()
        try:
            self._session.get(
                self._cfg.home_url,
                headers=self._headers(),
                timeout=self._cfg.timeout_s,
            ).raise_for_status()
        except requests.RequestException as exc:
            raise CrawlError("体彩主页预热失败") from exc
        self._warmed = True

    def _apply_rate_limit(self) -> None:
        """按 rate_limit_qps 在两次请求之间休眠。"""
        if self._cfg.rate_limit_qps <= 0:
            return
        elapsed = time.monotonic() - self._last_request
        wait_s = 1.0 / self._cfg.rate_limit_qps - elapsed
        if wait_s > 0:
            time.sleep(wait_s)
        self._last_request = time.monotonic()

    def _headers(self) -> dict[str, str]:
        """构造浏览器风格请求头；Referer/Origin 从 home_url 推导。"""
        parts = urlsplit(self._cfg.home_url)
        origin = f"{parts.scheme}://{parts.netloc}"
        return {
            "User-Agent": self._cfg.user_agent,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": self._cfg.accept_language,
            "Referer": self._cfg.home_url,
            "Origin": origin,
        }

    @staticmethod
    def _parse(payload: dict) -> tuple[list[SportsDraw], int]:
        """校验响应外层结构并解析每期记录。"""
        value = payload.get("value")
        if payload.get("success") is not True or not isinstance(value, dict):
            raise CrawlError("体彩接口返回失败状态")
        items = value.get("list")
        if not isinstance(items, list):
            raise CrawlError("体彩接口缺少开奖列表")
        try:
            pages = int(value.get("pages", 0))
        except (TypeError, ValueError) as exc:
            raise CrawlError("体彩接口页数非法") from exc
        return [SportteryCrawler._to_draw(item) for item in items], pages

    @staticmethod
    def _to_draw(item: dict) -> SportsDraw:
        """把单条官方记录转成 SportsDraw；字段缺失或格式错误转 CrawlError。"""
        try:
            draw = SportsDraw(
                issue=str(item["lotteryDrawNum"]).strip(),
                draw_date=date.fromisoformat(str(item["lotteryDrawTime"])[:10]),
                numbers=tuple(int(part) for part in str(item["lotteryDrawResult"]).split()),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CrawlError(f"体彩开奖记录无法解析: {item!r}") from exc
        return draw


class SportsStorage:
    """用 CSV 持久化体彩记录：期号幂等追加、原子重写。"""

    def __init__(self, path: Path) -> None:
        """记录 CSV 文件路径。"""
        self.path = path

    def load(self) -> pd.DataFrame:
        """读取全部记录；文件不存在或为空时返回同列空表。"""
        if not self.path.exists():
            return pd.DataFrame(columns=SPORTS_COLUMNS)
        try:
            return pd.read_csv(self.path, dtype={"issue": str})[SPORTS_COLUMNS]
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=SPORTS_COLUMNS)
        except (pd.errors.ParserError, KeyError, UnicodeDecodeError, OSError) as exc:
            raise StorageError(f"体彩数据文件无法读取: {self.path}") from exc

    def append(self, records: Sequence[SportsDraw]) -> int:
        """幂等追加新期号，按期号升序重写，返回新增行数。"""
        existing = self.load()
        known = set(existing["issue"])
        fresh = [record for record in records if not self._seen(record, known)]
        if not fresh:
            return 0
        rows = pd.DataFrame([self._to_row(record) for record in fresh], columns=SPORTS_COLUMNS)
        merged = pd.concat([existing, rows], ignore_index=True)
        self.rewrite(merged.sort_values("issue", kind="stable").reset_index(drop=True))
        return len(fresh)

    def rewrite(self, df: pd.DataFrame) -> None:
        """原子写回整表：先写临时文件，再替换目标文件。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(".csv.tmp")
        try:
            df.to_csv(tmp_path, index=False, columns=SPORTS_COLUMNS)
            os.replace(tmp_path, self.path)
        finally:
            tmp_path.unlink(missing_ok=True)

    @staticmethod
    def _seen(record: SportsDraw, known: set[str]) -> bool:
        """判断期号是否已存在，同时把新期号记入集合以过滤批内重复。"""
        if record.issue in known:
            return True
        known.add(record.issue)
        return False

    @staticmethod
    def _to_row(record: SportsDraw) -> dict[str, object]:
        """把一条记录转成 CSV 行，号码以空格连接。"""
        return {
            "issue": record.issue,
            "draw_date": record.draw_date.isoformat(),
            "numbers": " ".join(str(number) for number in record.numbers),
        }
