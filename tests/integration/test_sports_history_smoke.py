"""体彩历史数据 CLI 冒烟测试：采集器替换为内存假实现。"""

from datetime import date
from pathlib import Path

from lottery.cli.sports_history import main
from lottery.data.crawler import CrawlError
from lottery.data.sporttery import SportsDraw


class FakeCrawler:
    def __init__(self, _cfg):
        pass

    def fetch_all(self, _game_no):
        return [SportsDraw("26125", date(2026, 11, 2), (4, 4, 6))]


class FailingCrawler:
    def __init__(self, _cfg):
        pass

    def fetch_all(self, _game_no):
        raise CrawlError("接口不可用")


def test_cli_writes_selected_game(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("lottery.cli.sports_history.SportteryCrawler", FakeCrawler)
    monkeypatch.setattr(
        "lottery.cli.sports_history._storage_path",
        lambda game: tmp_path / Path(game.raw_csv).name,
    )
    assert main(["--game", "pl3"]) == 0
    assert "拉取 1 期，新增 1 期" in capsys.readouterr().out
    assert "numbers" in (tmp_path / "pl3_history.csv").read_text(encoding="utf-8")


def test_cli_reports_crawl_failure(monkeypatch, capsys):
    monkeypatch.setattr("lottery.cli.sports_history.SportteryCrawler", FailingCrawler)
    assert main(["--game", "dlt"]) == 1
    err = capsys.readouterr().err
    assert "接口不可用" in err
    assert "Traceback" not in err
