"""体育彩票采集与存储测试：使用假会话，不访问网络。"""

from datetime import date

import pytest
import requests

from lottery.common.config import AppConfig, get_config
from lottery.data.crawler import CrawlError
from lottery.data.sporttery import SportteryCrawler, SportsDraw, SportsStorage


class FakeResp:
    def __init__(self, status, payload, json_error=None):
        self.status_code = status
        self._payload = payload
        self._json_error = json_error

    def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


def _config() -> AppConfig:
    return get_config()


def _sporttery_config():
    cfg = _config().sporttery.model_copy(deep=True)
    cfg.rate_limit_qps = 0.0
    cfg.retry.backoff_s = 0.0
    return cfg


def _item(issue, day, result):
    return {
        "lotteryDrawNum": issue,
        "lotteryDrawTime": day,
        "lotteryDrawResult": result,
    }


def _payload(items, pages=1, success=True):
    return {"success": success, "value": {"pages": pages, "list": items}}


def test_fetch_page_parses_official_fields():
    session = FakeSession(
        [
            FakeResp(200, {}),
            FakeResp(200, _payload([_item("26125", "2026-11-02", "12 13 23 25 29 07 10")], 3)),
        ]
    )
    crawler = SportteryCrawler(_sporttery_config(), session)
    draws, pages = crawler.fetch_page("85", 1)
    assert pages == 3
    assert draws == [SportsDraw("26125", date(2026, 11, 2), (12, 13, 23, 25, 29, 7, 10))]
    assert session.calls[0][0] == _sporttery_config().home_url
    assert session.calls[1][1]["params"]["gameNo"] == "85"
    assert session.calls[1][1]["params"]["pageNo"] == 1


def test_success_false_raises_crawl_error():
    session = FakeSession([FakeResp(200, {}), FakeResp(200, _payload([], success=False))])
    with pytest.raises(CrawlError):
        SportteryCrawler(_sporttery_config(), session).fetch_page("85", 1)


def test_bad_record_raises_crawl_error():
    session = FakeSession(
        [FakeResp(200, {}), FakeResp(200, _payload([{"lotteryDrawNum": "26125"}]))]
    )
    with pytest.raises(CrawlError):
        SportteryCrawler(_sporttery_config(), session).fetch_page("85", 1)


def test_html_response_raises_crawl_error():
    session = FakeSession([FakeResp(200, {}), FakeResp(200, {}, json_error=ValueError("html"))])
    with pytest.raises(CrawlError):
        SportteryCrawler(_sporttery_config(), session).fetch_page("85", 1)


def test_fetch_all_follows_reported_page_count():
    session = FakeSession(
        [
            FakeResp(200, {}),
            FakeResp(
                200,
                _payload(
                    [
                        _item("26125", "2026-11-02", "4 4 6"),
                        _item("26124", "2026-11-01", "1 2 3"),
                    ],
                    2,
                ),
            ),
            FakeResp(200, _payload([_item("26123", "2026-10-31", "7 8 9")], 2)),
        ]
    )
    draws = SportteryCrawler(_sporttery_config(), session).fetch_all("35")
    assert [draw.issue for draw in draws] == ["26125", "26124", "26123"]
    assert [call[1]["params"]["pageNo"] for call in session.calls[1:]] == [1, 2]


def test_fetch_all_respects_max_pages():
    cfg = _sporttery_config()
    cfg.max_pages = 1
    session = FakeSession(
        [
            FakeResp(200, {}),
            FakeResp(200, _payload([_item("26125", "2026-11-02", "4 4 6")], 5)),
        ]
    )
    draws = SportteryCrawler(cfg, session).fetch_all("35")
    assert len(draws) == 1
    assert len(session.calls) == 2


def test_retries_then_raises_crawl_error():
    cfg = _sporttery_config()
    cfg.retry.max_attempts = 2
    session = FakeSession([FakeResp(200, {}), FakeResp(500, {}), FakeResp(500, {})])
    with pytest.raises(CrawlError):
        SportteryCrawler(cfg, session).fetch_page("85", 1)
    assert len(session.calls) == 3


def test_storage_append_is_idempotent_and_sorted(tmp_path):
    storage = SportsStorage(tmp_path / "pl3_history.csv")
    assert storage.load().empty
    later = SportsDraw("26125", date(2026, 11, 2), (4, 4, 6))
    earlier = SportsDraw("26124", date(2026, 11, 1), (1, 2, 3))
    assert storage.append([later, earlier]) == 2
    assert storage.append([later]) == 0
    loaded = storage.load()
    assert list(loaded["issue"]) == ["26124", "26125"]
    assert loaded.loc[1, "numbers"] == "4 4 6"
