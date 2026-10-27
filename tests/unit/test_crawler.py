import pytest
import requests

from lottery.common.config import get_config
from lottery.data.crawler import SSQCrawler, CrawlError


class FakeResp:
    def __init__(self, status, payload):
        self.status_code, self._p = status, payload

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, **kw):
        self.calls.append((url, kw))
        return self.responses.pop(0)


def payload(*codes):
    return {"state": 0, "result": [{"code": c, "date": "2026-10-25(日)",
            "red": "01,02,03,04,05,06", "blue": "09"} for c in codes]}


def test_fetch_page_parses():
    s = FakeSession([FakeResp(200, payload("2026122"))])
    out = SSQCrawler(get_config().crawler, s).fetch_page(1)
    assert out[0].issue == "2026122"
    assert s.calls[0][0].endswith("findDrawNotice")


def test_state_error_raises():
    s = FakeSession([FakeResp(200, {"state": 1, "result": []})])
    with pytest.raises(CrawlError):
        SSQCrawler(get_config().crawler, s).fetch_page(1)


def test_non_dict_json_raises_crawl_error():
    s = FakeSession([FakeResp(200, [])])
    with pytest.raises(CrawlError):
        SSQCrawler(get_config().crawler, s).fetch_page(1)
    assert len(s.calls) == 1


def test_retries_then_raises():
    cfg = get_config().crawler.model_copy(deep=True)
    cfg.retry.max_attempts = 2
    cfg.retry.backoff_s = 0.0
    s = FakeSession([FakeResp(500, {}), FakeResp(500, {})])
    with pytest.raises(CrawlError):
        SSQCrawler(cfg, s).fetch_page(1)
    assert len(s.calls) == 2


def test_fetch_all_stops_on_short_page():
    cfg = get_config().crawler.model_copy(deep=True)
    cfg.page_size = 2
    cfg.rate_limit_qps = 0.0
    s = FakeSession([FakeResp(200, payload("2026122", "2026121")),
                     FakeResp(200, payload("2026120"))])
    pages = list(SSQCrawler(cfg, s).fetch_all())
    assert [len(p) for p in pages] == [2, 1]


def test_fetch_all_respects_max_pages():
    cfg = get_config().crawler.model_copy(deep=True)
    cfg.page_size = 2
    cfg.rate_limit_qps = 0.0
    s = FakeSession([FakeResp(200, payload("2026122", "2026121")),
                     FakeResp(200, payload("2026120", "2026119"))])
    pages = list(SSQCrawler(cfg, s).fetch_all(max_pages=2))
    assert [len(p) for p in pages] == [2, 2]
    assert len(s.calls) == 2


def test_fetch_since_stops_at_known_issue():
    cfg = get_config().crawler.model_copy(deep=True)
    cfg.page_size = 2
    cfg.rate_limit_qps = 0.0
    s = FakeSession([FakeResp(200, payload("2026122", "2026121"))])
    out = SSQCrawler(cfg, s).fetch_since("2026121")
    assert [d.issue for d in out] == ["2026122"]
