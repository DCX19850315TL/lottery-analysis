import itertools

import pytest

from lottery.theory.bounds import covering_lower_bound
from lottery.theory.coverage import CoverageAlgorithm, CoverageDesigner, CoveragePlan, Guarantee


@pytest.mark.parametrize(
    ("guarantee", "t"),
    [
        (Guarantee.MATCH6_COVER5, 5),
        (Guarantee.MATCH5_COVER4, 4),
        (Guarantee.MATCH4_COVER3, 3),
    ],
)
def test_guarantee_maps_to_t(guarantee, t):
    assert guarantee.t == t


@pytest.mark.parametrize("algorithm", list(CoverageAlgorithm))
def test_plan_tickets_come_from_pool(algorithm):
    pool = tuple(range(1, 9))
    plan = CoverageDesigner(algorithm).build(pool, Guarantee.MATCH6_COVER5)
    assert isinstance(plan, CoveragePlan)
    assert plan.guarantee is Guarantee.MATCH6_COVER5
    assert len({ticket.reds for ticket in plan.tickets}) == len(plan.tickets)
    for ticket in plan.tickets:
        assert len(ticket.reds) == 6
        assert ticket.reds == tuple(sorted(ticket.reds))
        assert set(ticket.reds) <= set(pool)


@pytest.mark.parametrize("algorithm", list(CoverageAlgorithm))
def test_plan_covers_every_t_subset(algorithm):
    pool = tuple(range(1, 9))
    plan = CoverageDesigner(algorithm).build(pool, Guarantee.MATCH6_COVER5)
    covered = {tuple(c) for ticket in plan.tickets for c in itertools.combinations(ticket.reds, 5)}
    for target in itertools.combinations(pool, 5):
        assert target in covered


@pytest.mark.parametrize("algorithm", list(CoverageAlgorithm))
def test_plan_bound_and_optimal_flag(algorithm):
    pool = tuple(range(1, 9))
    plan = CoverageDesigner(algorithm).build(pool, Guarantee.MATCH6_COVER5)
    assert plan.lower_bound == covering_lower_bound(8, 6, 5)
    assert len(plan.tickets) >= plan.lower_bound
    assert plan.proven_optimal is (len(plan.tickets) == plan.lower_bound)
    assert plan.cost == float(len(plan.tickets))


@pytest.mark.parametrize("algorithm", list(CoverageAlgorithm))
def test_build_is_deterministic(algorithm):
    builder = CoverageDesigner(algorithm)
    first = builder.build(tuple(range(1, 9)), Guarantee.MATCH5_COVER4)
    second = builder.build(tuple(range(1, 9)), Guarantee.MATCH5_COVER4)
    assert first == second


def test_single_block_is_proven_optimal():
    plan = CoverageDesigner(CoverageAlgorithm.GREEDY).build(tuple(range(1, 7)), Guarantee.MATCH6_COVER5)
    assert len(plan.tickets) == 1
    assert plan.lower_bound == 1
    assert plan.proven_optimal


def test_greedy_result_has_no_redundant_block():
    plan = CoverageDesigner(CoverageAlgorithm.GREEDY).build(tuple(range(1, 11)), Guarantee.MATCH6_COVER5)
    for index, ticket in enumerate(plan.tickets):
        others = [t for position, t in enumerate(plan.tickets) if position != index]
        covered = {tuple(c) for t in others for c in itertools.combinations(t.reds, 5)}
        assert any(tuple(c) not in covered for c in itertools.combinations(ticket.reds, 5))


@pytest.mark.parametrize("pool", [tuple(range(1, 6)), (1, 2, 2, 3, 4, 5, 6)])
def test_invalid_pool_rejected(pool):
    with pytest.raises(ValueError):
        CoverageDesigner(CoverageAlgorithm.GREEDY).build(pool, Guarantee.MATCH6_COVER5)
