import itertools

from lottery.cli.theory_report import build_report, main
from lottery.common.types import MathOptions
from lottery.theory.coverage import Guarantee


def _options(pool_size=10, guarantee="match4_cover3", seed=7):
    return MathOptions(
        pool_size=pool_size,
        guarantee=guarantee,
        algorithm="greedy",
        seed=seed,
        time_limit_s=10.0,
    )


def test_build_report_is_deterministic_and_uses_pool_size_numbers():
    first = build_report(_options())
    second = build_report(_options())
    assert first.pool_numbers == second.pool_numbers
    assert first.plan == second.plan
    assert len(first.pool_numbers) == 10
    assert list(first.pool_numbers) == sorted(set(first.pool_numbers))
    assert all(1 <= number <= 33 for number in first.pool_numbers)


def test_plan_covers_every_three_subset():
    report = build_report(_options())
    ticket_sets = [set(ticket.reds) for ticket in report.plan.tickets]
    for subset in itertools.combinations(report.pool_numbers, 3):
        assert any(set(subset) <= ticket for ticket in ticket_sets)


def test_table_reference_reports_lower_bound_status():
    report = build_report(_options())
    assert report.table_reference is None  # C(10,6,3) 未收录，走构造算法
    assert report.plan.guarantee is Guarantee.MATCH4_COVER3
    assert report.plan.lower_bound >= 1


def test_cli_prints_banner_and_plan(capsys):
    code = main(["--pool-size", "10", "--guarantee", "match4_cover3", "--seed", "7"])
    out = capsys.readouterr().out
    assert code == 0
    assert "风险提示" in out
    assert "覆盖方案" in out
    assert "中4保3" in out
    assert "号码池: 10" in out


def test_cli_rejects_small_pool():
    assert main(["--pool-size", "4"]) == 2
