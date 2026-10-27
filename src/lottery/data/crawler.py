"""福彩接口分页采集：带重试、限速与显式失败。"""

import time
from collections.abc import Iterator

import requests

from lottery.common.config import CrawlerConfig
from lottery.common.types import DrawRecord
from lottery.data.parser import ParseError, parse_response


class CrawlError(RuntimeError):
    """采集失败：请求重试耗尽或接口响应无法解析。"""


class SSQCrawler:
    """按页采集双色球历史开奖记录。"""

    def __init__(self, cfg: CrawlerConfig, session: requests.Session | None = None) -> None:
        """保存采集配置；未传入会话时新建 requests.Session。"""
        self._cfg = cfg
        self._session = session if session is not None else requests.Session()

    def _apply_rate_limit(self) -> None:
        """成功请求后按 qps 休眠；qps <= 0 时关闭限速。"""
        if self._cfg.rate_limit_qps > 0:
            time.sleep(1 / self._cfg.rate_limit_qps)

    def _request_once(self, page: int) -> dict[str, object]:
        """发起一次分页请求并返回 JSON 对象，状态或格式异常直接抛出。"""
        params = {"name": "ssq", "pageNo": page,
                  "pageSize": self._cfg.page_size, "systemType": "PC"}
        response = self._session.get(
            self._cfg.api_url,
            params=params,
            headers={"User-Agent": self._cfg.user_agent},
            timeout=self._cfg.timeout_s,
        )
        response.raise_for_status()
        if response.status_code != 200:
            raise CrawlError(f"接口返回状态码 {response.status_code}")
        self._apply_rate_limit()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ParseError(f"接口响应不是对象: {type(payload).__name__}")
        return payload

    def _request_with_retry(self, page: int) -> dict[str, object]:
        """请求一页 JSON，传输或状态失败按配置重试；耗尽后抛 CrawlError。"""
        attempts = max(1, self._cfg.retry.max_attempts)
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                return self._request_once(page)
            except (requests.RequestException, CrawlError) as exc:
                last_exc = exc
                if attempt + 1 < attempts:
                    time.sleep(self._cfg.retry.backoff_s * 2**attempt)
        raise CrawlError(f"采集第 {page} 页失败，已尝试 {attempts} 次") from last_exc

    def _load_page(self, page: int) -> list[DrawRecord]:
        """请求并解析一页；解析失败立即包装为 CrawlError。"""
        payload = self._request_with_retry(page)
        try:
            return parse_response(payload)
        except ParseError as exc:
            raise CrawlError(f"第 {page} 页响应解析失败: {exc}") from exc

    def _page_limit(self, max_pages: int | None) -> int:
        """返回页数上限：显式传入优先，否则用配置的 max_pages。"""
        return self._cfg.max_pages if max_pages is None else max_pages

    def fetch_page(self, page: int) -> list[DrawRecord]:
        """采集并解析单页，失败时抛 CrawlError。"""
        return self._load_page(page)

    def fetch_all(self, max_pages: int | None = None) -> Iterator[list[DrawRecord]]:
        """逐页产出开奖记录；空页、短页或达到页数上限时停止。"""
        for page in range(1, self._page_limit(max_pages) + 1):
            records = self._load_page(page)
            if not records:
                return
            yield records
            if len(records) < self._cfg.page_size:
                return

    def fetch_since(self, issue: str) -> list[DrawRecord]:
        """按 API 顺序采集，返回 issue 之前的新记录；遇已有期号即截断。"""
        collected: list[DrawRecord] = []
        for records in self.fetch_all():
            for record in records:
                if record.issue <= issue:
                    return collected
                collected.append(record)
        return collected
