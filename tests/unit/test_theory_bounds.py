import math

import pytest
from hypothesis import given, settings, strategies as st

from lottery.theory.bounds import covering_lower_bound, schoenheim_bound, sphere_covering_bound
from lottery.theory.coverage_tables import CoverageTable


def test_sphere_bound_matches_closed_form():
    assert sphere_covering_bound(33, 6, 5) == math.ceil(math.comb(33, 5) / math.comb(6, 5))
    assert sphere_covering_bound(10, 4, 3) == math.ceil(math.comb(10, 3) / math.comb(4, 3))


def test_schoenheim_recursion_small_instance():
    # L(9,3,1)=3 -> L(10,4,2)=ceil(10/4*3)=8 -> L(11,5,3)=ceil(11/5*8)=18 -> L(12,6,4)=36
    assert schoenheim_bound(12, 6, 4) == 36


def test_boundary_cases():
    assert covering_lower_bound(33, 6, 0) == 1
    assert covering_lower_bound(33, 6, 6) == math.comb(33, 6)
    assert covering_lower_bound(33, 6, 1) == math.ceil(33 / 6)


def test_covering_lower_bound_is_max_of_two_bounds():
    for v, k, t in [(10, 6, 2), (12, 6, 3), (20, 6, 4)]:
        expected = max(sphere_covering_bound(v, k, t), schoenheim_bound(v, k, t))
        assert covering_lower_bound(v, k, t) == expected


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        covering_lower_bound(5, 6, 2)
    with pytest.raises(ValueError):
        covering_lower_bound(10, 6, 7)
    with pytest.raises(ValueError):
        covering_lower_bound(10, 0, 1)


@given(st.integers(1, 40), st.integers(1, 8))
@settings(max_examples=50, deadline=None)
def test_t1_bound_is_exact(v, k):
    if k > v:
        return
    assert covering_lower_bound(v, k, 1) == math.ceil(v / k)


def test_table_lookup_t1_is_exact():
    assert CoverageTable().lookup(33, 6, 1) == (math.ceil(33 / 6), True)


def test_table_lookup_full_t_needs_every_block():
    assert CoverageTable().lookup(33, 6, 6) == (math.comb(33, 6), True)


def test_table_lookup_miss_returns_none():
    assert CoverageTable().lookup(33, 6, 5) is None


def test_table_rejects_entry_below_lower_bound():
    with pytest.raises(ValueError):
        CoverageTable({(33, 6, 5): 3})


def test_table_marks_entry_above_bound_as_not_proven():
    bound = covering_lower_bound(33, 6, 5)
    table = CoverageTable({(33, 6, 5): bound + 1})
    assert table.lookup(33, 6, 5) == (bound + 1, False)


def test_table_marks_entry_at_bound_as_proven():
    bound = covering_lower_bound(33, 6, 5)
    table = CoverageTable({(33, 6, 5): bound})
    assert table.lookup(33, 6, 5) == (bound, True)
