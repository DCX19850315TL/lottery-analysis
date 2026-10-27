from datetime import date
import pandas as pd
from lottery.common.types import DrawRecord
from lottery.data.cleaner import DrawCleaner
from lottery.data.repository import DrawRepository
from lottery.data.storage import DrawStorage, COLUMNS

def rec(i): return DrawRecord(i, date(2026, 10, 25), (1, 2, 3, 4, 5, 6), 9)

class FakeCrawler:
    def __init__(self, full=(), since=()): self.full, self.since_out = list(full), list(since)
    def fetch_all(self, max_pages=None): yield self.full
    def fetch_since(self, issue): return self.since_out

def repo(tmp_path, crawler):
    return DrawRepository(DrawStorage(tmp_path / "x.csv"), DrawCleaner(), crawler)

def test_refresh_full_when_empty(tmp_path):
    r = repo(tmp_path, FakeCrawler(full=[rec("2026002"), rec("2026001")]))
    res = r.refresh()
    assert [d.issue for d in r.all_draws()] == ["2026001", "2026002"]
    assert res.ok is True

def test_refresh_incremental_appends(tmp_path):
    storage = DrawStorage(tmp_path / "x.csv")
    DrawRepository(storage, DrawCleaner(), FakeCrawler(full=[rec("2026001")])).refresh()
    r2 = DrawRepository(storage, DrawCleaner(), FakeCrawler(since=[rec("2026002")]))
    r2.refresh()
    assert r2.latest_issue() == "2026002"
    assert len(r2.all_draws()) == 2

def test_failed_full_crawl_leaves_csv_untouched(tmp_path):
    class Boom:
        def fetch_all(self, max_pages=None): raise RuntimeError("boom"); yield
        def fetch_since(self, issue): raise RuntimeError("boom")
    r = repo(tmp_path, Boom())
    try:
        r.refresh()
    except RuntimeError:
        pass
    assert not (tmp_path / "x.csv").exists()

def test_slice_is_inclusive(tmp_path):
    r = repo(tmp_path, FakeCrawler(full=[rec("2026001"), rec("2026002"), rec("2026003")]))
    r.refresh()
    assert [d.issue for d in r.slice("2026002", "2026003")] == ["2026002", "2026003"]
