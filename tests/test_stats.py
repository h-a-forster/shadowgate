from __future__ import annotations

import math
import random

import pytest

from shadowgate import stats
from shadowgate.errors import InsufficientData
from shadowgate.stats import (
    Bin,
    Estimate,
    aurc,
    auroc,
    bootstrap_ci,
    brier,
    clopper_pearson,
    ece,
    mcnemar_p,
    mean_ci,
    paired_diff,
    regularized_beta,
    reliability,
    required_n,
    risk_coverage,
    weighted_proportion,
    wilson,
    z_for,
)

# ---------------------------------------------------------------------------- z_for


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        (0.95, 1.959963984540054),
        (0.99, 2.5758293035489),
        (0.90, 1.6448536269514722),
        (0.6826894921370859, 1.0),
        (0.999, 3.2905267314919255),
    ],
)
def test_z_for_known_values(level, expected):
    assert z_for(level) == pytest.approx(expected, abs=1e-12)


def test_z_for_extreme_tail_roundtrip():
    for level in (1e-6, 0.5, 0.999999, 1 - 1e-12):
        z = z_for(level)
        tail = 0.5 * math.erfc(z / math.sqrt(2))
        assert tail == pytest.approx((1 - level) / 2, rel=1e-9)


def test_norm_ppf_symmetry_and_regions():
    for p in (1e-10, 0.01, 0.02425, 0.3, 0.5, 0.7, 0.98, 1 - 1e-10):
        x = stats._norm_ppf(p)
        assert 0.5 * math.erfc(-x / math.sqrt(2)) == pytest.approx(p, rel=1e-9)
        assert stats._norm_ppf(1 - p) == pytest.approx(-x, abs=1e-6)


@pytest.mark.parametrize("bad", [0, 1, 1.5, -0.1, float("nan"), "0.95", True])
def test_z_for_rejects_bad_level(bad):
    with pytest.raises(ValueError):
        z_for(bad)


# ---------------------------------------------------------------------------- wilson


def test_wilson_reference_values():
    # Closed-form Wilson bounds evaluated by hand (z = 1.959964)
    e = wilson(5, 10)
    assert e.value == 0.5
    assert e.lo == pytest.approx(0.2365931, abs=1e-6)
    assert e.hi == pytest.approx(0.7634069, abs=1e-6)
    e = wilson(1, 20)
    assert e.lo == pytest.approx(0.0088814, abs=1e-6)
    assert e.hi == pytest.approx(0.2361311, abs=1e-6)


def test_wilson_zero_and_full():
    e0 = wilson(0, 10)
    assert e0.lo == 0.0 and e0.value == 0.0
    assert e0.hi == pytest.approx(0.2775328, abs=1e-6)
    e1 = wilson(10, 10)
    assert e1.hi == 1.0
    assert e1.lo == pytest.approx(1 - 0.2775328, abs=1e-6)


def test_wilson_empty_is_vacuous():
    e = wilson(0, 0)
    assert e == Estimate(None, 0.0, 1.0, 0, "wilson", 0.95)


def test_wilson_level_and_method():
    e90, e99 = wilson(3, 30, 0.90), wilson(3, 30, 0.99)
    assert e90.hi - e90.lo < e99.hi - e99.lo
    assert e90.method == "wilson" and e99.level == 0.99


def test_wilson_bounds_contain_estimate():
    for n in (1, 2, 7, 50):
        for k in range(n + 1):
            e = wilson(k, n)
            assert 0.0 <= e.lo <= e.value <= e.hi <= 1.0


@pytest.mark.parametrize(("k", "n"), [(-1, 5), (6, 5), (1.0, 5), (1, 5.0), (True, 5)])
def test_wilson_rejects_bad_counts(k, n):
    with pytest.raises(ValueError):
        wilson(k, n)


# ---------------------------------------------------------------------------- incomplete beta


def test_regularized_beta_known():
    assert regularized_beta(0.5, 2, 3) == pytest.approx(0.6875, abs=1e-14)
    assert regularized_beta(0.3, 1, 1) == pytest.approx(0.3, abs=1e-14)
    # I_x(a, 1) = x^a
    assert regularized_beta(0.7, 4.5, 1) == pytest.approx(0.7**4.5, rel=1e-12)
    assert regularized_beta(0.0, 2, 2) == 0.0
    assert regularized_beta(1.0, 2, 2) == 1.0


def test_regularized_beta_symmetry():
    for x, a, b in [(0.2, 3, 7), (0.9, 50, 2), (0.45, 120, 130)]:
        assert regularized_beta(x, a, b) + regularized_beta(1 - x, b, a) == pytest.approx(1.0)


def test_regularized_beta_rejects_bad_input():
    with pytest.raises(ValueError):
        regularized_beta(0.5, 0, 1)
    with pytest.raises(ValueError):
        regularized_beta(1.5, 1, 1)


# ---------------------------------------------------------------------------- clopper_pearson


@pytest.mark.parametrize(
    ("k", "n", "lo", "hi"),
    [
        (0, 10, 0.0, 0.3084971),
        (5, 10, 0.187086, 0.812914),
        (1, 20, 0.001265, 0.248733),
        (10, 10, 0.6915029, 1.0),
        (3, 100, 0.0062300, 0.0851761),
    ],
)
def test_clopper_pearson_reference(k, n, lo, hi):
    e = clopper_pearson(k, n)
    assert e.lo == pytest.approx(lo, abs=5e-6)
    assert e.hi == pytest.approx(hi, abs=5e-6)
    assert e.value == k / n
    assert e.method == "clopper-pearson"


def test_clopper_pearson_edges_exact():
    e = clopper_pearson(0, 10)
    assert e.lo == 0.0
    assert e.hi == pytest.approx(1 - 0.025**0.1, abs=1e-15)
    e = clopper_pearson(10, 10)
    assert e.hi == 1.0


def test_clopper_pearson_wider_than_wilson():
    for k, n in [(1, 20), (5, 10), (30, 100)]:
        cp, w = clopper_pearson(k, n), wilson(k, n)
        assert cp.lo <= w.lo + 1e-12 and cp.hi >= w.hi - 1e-12


def test_clopper_pearson_tail_identity():
    # P[X >= k | p = lo] = alpha/2  <=>  I_lo(k, n-k+1) = alpha/2
    e = clopper_pearson(7, 40, 0.9)
    assert regularized_beta(e.lo, 7, 34) == pytest.approx(0.05, abs=1e-9)
    assert regularized_beta(e.hi, 8, 33) == pytest.approx(0.95, abs=1e-9)


def test_clopper_pearson_empty_and_invalid():
    assert clopper_pearson(0, 0) == Estimate(None, 0.0, 1.0, 0, "clopper-pearson", 0.95)
    with pytest.raises(ValueError):
        clopper_pearson(3, 2)
    with pytest.raises(ValueError):
        clopper_pearson(1, 2, level=1.0)


# ---------------------------------------------------------------------------- weighted_proportion


def test_weighted_equal_weights_match_clopper_pearson_exactly():
    for k, n in [(0, 5), (3, 10), (10, 10), (17, 123)]:
        ys = [True] * k + [False] * (n - k)
        for w in (1.0, 2.5, 7):
            e = weighted_proportion(ys, [w] * n)
            ref = clopper_pearson(k, n)
            assert (e.value, e.lo, e.hi) == (ref.value, ref.lo, ref.hi)
            assert e.n == n and e.n_eff == n
            assert e.method == "hajek-korn-graubard"


def test_weighted_equal_weights_wilson_method_matches_wilson_exactly():
    for k, n in [(0, 5), (3, 10), (10, 10), (17, 123)]:
        ys = [True] * k + [False] * (n - k)
        for w in (1.0, 2.5, 7):
            e = weighted_proportion(ys, [w] * n, method="wilson")
            ref = wilson(k, n)
            assert (e.value, e.lo, e.hi) == (ref.value, ref.lo, ref.hi)
            assert e.n == n and e.n_eff == n
            assert e.method == "hajek-wilson"


def test_weighted_rejects_unknown_method():
    with pytest.raises(ValueError):
        weighted_proportion([True, False], [1.0, 2.0], method="wald")


def test_korn_graubard_fractional_counts_match_beta_quantiles():
    # Reference values: scipy.stats.beta.ppf(0.025, x, n - x + 1) and
    # beta.ppf(0.975, x + 1, n - x) with x = p * n_eff (non-integer).
    lo, hi = stats.weighted_bounds(0.13, 23.7, 0.95, "korn-graubard")
    x, n = 0.13 * 23.7, 23.7
    assert stats.regularized_beta(lo, x, n - x + 1) == pytest.approx(0.025, abs=1e-12)
    assert stats.regularized_beta(hi, x + 1, n - x) == pytest.approx(0.975, abs=1e-12)
    assert 0.0 < lo < 0.13 < hi < 1.0


def test_korn_graubard_wider_than_wilson_for_weighted_sample():
    ys = [True, False, False, True, False, False]
    ws = [1.0, 1.0, 10.0, 3.0, 2.0, 1.0]
    kg = weighted_proportion(ys, ws)
    wi = weighted_proportion(ys, ws, method="wilson")
    assert kg.value == wi.value and kg.n_eff == wi.n_eff
    assert kg.lo <= wi.lo and kg.hi >= wi.hi


def test_korn_graubard_coverage_smoke_low_pi_stratum():
    # Reviewer's design: half the population sits in a pi = .02 stratum holding most errors.
    # Hajek-Wilson covers ~0.87 here at N = 500 and Korn-Graubard ~0.90 (still below nominal:
    # this is the sparse-stratum case in which audit.summarize refuses to report "ok").
    rng = random.Random(4)
    design = [(100, 0.5, 0.02), (150, 0.1, 0.02), (250, 0.02, 0.40)]
    reps, kg_cov, wi_cov, used = 600, 0, 0, 0
    for _ in range(reps):
        ys, ws, k, t = [], [], 0, 0
        for m, p, e in design:
            for _ in range(m):
                y = rng.random() < e
                t += 1
                k += y
                if rng.random() < p:
                    ys.append(y)
                    ws.append(1.0 / p)
        true = k / t
        kg = weighted_proportion(ys, ws)
        wi = weighted_proportion(ys, ws, method="wilson")
        used += 1
        kg_cov += kg.lo <= true <= kg.hi
        wi_cov += wi.lo <= true <= wi.hi
    assert wi_cov / used < 0.89
    assert kg_cov / used >= 0.89
    assert kg_cov - wi_cov >= 0.02 * used


def test_weighted_hajek_point_estimate():
    ys = [True, False, False, True]
    ws = [1.0, 1.0, 4.0, 2.0]
    e = weighted_proportion(ys, ws)
    assert e.value == pytest.approx(3.0 / 8.0)
    assert 0.0 <= e.lo < e.value < e.hi <= 1.0


def test_weighted_n_eff_is_min_of_kish_and_linearised():
    ys = [True, False, False, True, False, False]
    ws = [1.0, 1.0, 10.0, 3.0, 2.0, 1.0]
    sw, sw2 = sum(ws), sum(w * w for w in ws)
    p = (1.0 + 3.0) / sw
    var = sum(w * w * (y - p) ** 2 for w, y in zip(ws, ys, strict=True)) / sw**2
    expected = min(sw * sw / sw2, p * (1 - p) / var)
    e = weighted_proportion(ys, ws)
    assert e.n_eff == pytest.approx(expected)
    lo, hi = stats._cp_bounds(p * expected, expected, 0.95)
    assert (e.lo, e.hi) == (pytest.approx(lo), pytest.approx(hi))
    w = weighted_proportion(ys, ws, method="wilson")
    lo, hi = stats._wilson_bounds(p, expected, z_for(0.95))
    assert (w.lo, w.hi) == (pytest.approx(lo), pytest.approx(hi))


def test_weighted_errors_in_heavy_units_widen_interval():
    # Same Kish n_eff, but errors sit on the heavily weighted units in the second case.
    ws = [1.0] * 18 + [9.0, 9.0]
    light = weighted_proportion([True, True] + [False] * 18, ws)
    heavy = weighted_proportion([False] * 18 + [True, True], ws)
    kish = sum(ws) ** 2 / sum(w * w for w in ws)
    assert light.n_eff == pytest.approx(kish)
    assert heavy.n_eff < kish


def test_weighted_all_zero_or_all_one():
    e0 = weighted_proportion([False] * 4, [1.0, 2.0, 3.0, 4.0])
    assert e0.value == 0.0 and e0.lo == 0.0 and e0.hi < 1.0
    assert e0.n_eff == pytest.approx(100 / 30)
    e1 = weighted_proportion([True] * 4, [1.0, 2.0, 3.0, 4.0])
    assert e1.value == 1.0 and e1.hi == 1.0


def test_weighted_empty():
    e = weighted_proportion([], [])
    assert e.value is None and (e.lo, e.hi) == (0.0, 1.0) and e.n == 0


@pytest.mark.parametrize(
    "ws",
    [[1.0, 0.0], [1.0, -2.0], [1.0, float("nan")], [1.0, float("inf")], [1.0]],
)
def test_weighted_rejects_bad_weights(ws):
    with pytest.raises(ValueError):
        weighted_proportion([True, False], ws)


def test_weighted_rejects_bad_labels():
    with pytest.raises(ValueError):
        weighted_proportion([True, 2], [1.0, 1.0])


def test_weighted_accepts_int_labels():
    assert weighted_proportion([1, 0], [1.0, 3.0]).value == pytest.approx(0.25)


def _simulate_population(rng: random.Random, n: int):
    conf = [rng.random() for _ in range(n)]
    # Error probability strongly confidence-dependent; non-monotone so neither end dominates.
    err = [rng.random() < 0.45 * (1 - c) ** 2 + 0.12 * c for c in conf]
    pi = [0.15 if c < 0.5 else 0.05 if c < 0.8 else 0.02 for c in conf]
    return err, pi


def test_weighted_monte_carlo_bias_and_coverage():
    rng = random.Random(20240611)
    err, pi = _simulate_population(rng, 6000)
    true = sum(err) / len(err)
    reps = 400
    hajek, naive, covered = [], [], 0
    for _ in range(reps):
        ys, ws = [], []
        for e, p in zip(err, pi, strict=True):
            if rng.random() < p:
                ys.append(e)
                ws.append(1.0 / p)
        est = weighted_proportion(ys, ws)
        hajek.append(est.value)
        naive.append(sum(ys) / len(ys))
        covered += est.lo <= true <= est.hi
    mean_hajek = sum(hajek) / reps
    mean_naive = sum(naive) / reps
    # (a) Hajek approximately unbiased; naive unweighted mean clearly biased.
    assert abs(mean_hajek - true) < 0.005
    assert abs(mean_naive - true) > 0.03
    assert abs(mean_naive - true) > 5 * abs(mean_hajek - true)
    # (b) nominal 95% interval covers the true rate in roughly 90-99% of replications.
    coverage = covered / reps
    assert 0.90 <= coverage <= 0.99, coverage


# ---------------------------------------------------------------------------- required_n


def test_required_n_values():
    assert required_n(0.5, 0.05) == 385
    assert required_n(0.1, 0.03) == math.ceil(z_for(0.95) ** 2 * 0.09 / 0.0009)
    assert required_n(0.5, 0.05, level=0.99) > required_n(0.5, 0.05)
    assert required_n(0.0, 0.05) == 1


@pytest.mark.parametrize(
    ("p", "h"), [(-0.1, 0.05), (1.1, 0.05), (0.5, 0.0), (0.5, 1.0), (float("nan"), 0.05)]
)
def test_required_n_validation(p, h):
    with pytest.raises(ValueError):
        required_n(p, h)


# ---------------------------------------------------------------------------- mean_ci


def test_mean_ci_values():
    xs = [1.0, 2.0, 3.0, 4.0]
    e = mean_ci(xs)
    s = math.sqrt(sum((x - 2.5) ** 2 for x in xs) / 3)
    half = 1.959963984540054 * s / 2
    assert e.value == 2.5
    assert (e.lo, e.hi) == (pytest.approx(2.5 - half), pytest.approx(2.5 + half))
    assert e.n == 4 and e.method == "normal"


def test_mean_ci_empty_and_single():
    assert mean_ci([]) == Estimate(None, None, None, 0, "normal", 0.95)
    e = mean_ci([3.0])
    assert e.value == 3.0 and e.lo is None and e.hi is None


def test_mean_ci_constant_and_invalid():
    e = mean_ci([2.0, 2.0, 2.0])
    assert e.lo == e.hi == 2.0
    with pytest.raises(ValueError):
        mean_ci([1.0, float("nan")])


# ---------------------------------------------------------------------------- bootstrap


def test_bootstrap_deterministic_and_reasonable():
    xs = [float(i) for i in range(50)]

    def stat(idx):
        return sum(xs[i] for i in idx) / len(idx)

    a = bootstrap_ci(stat, 50, reps=500, seed=1)
    b = bootstrap_ci(stat, 50, reps=500, seed=1)
    c = bootstrap_ci(stat, 50, reps=500, seed=2)
    assert a == b and a != c
    lo, hi = a
    assert lo < 24.5 < hi
    normal = mean_ci(xs)
    assert lo == pytest.approx(normal.lo, abs=1.5) and hi == pytest.approx(normal.hi, abs=1.5)


def test_bootstrap_validation():
    with pytest.raises(InsufficientData):
        bootstrap_ci(lambda idx: 0.0, 0)
    with pytest.raises(ValueError):
        bootstrap_ci(lambda idx: 0.0, 5, reps=0)
    with pytest.raises(ValueError):
        bootstrap_ci(lambda idx: float("nan"), 5, reps=3)


def test_quantile_type7():
    assert stats._quantile([0.0, 10.0], 0.25) == 2.5
    assert stats._quantile([1.0, 2.0, 3.0], 1.0) == 3.0


# ---------------------------------------------------------------------------- paired_diff


def test_paired_diff_basic():
    a = [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0]
    b = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    e = paired_diff(a, b, reps=1000, seed=3)
    assert e.value == pytest.approx(3 / 8)
    assert e.lo <= e.value <= e.hi
    assert e.method == "paired-bootstrap" and e.n == 8
    assert paired_diff(a, b, reps=1000, seed=3) == e


def test_paired_diff_constant_difference():
    e = paired_diff([2.0, 3.0, 4.0], [1.0, 2.0, 3.0])
    assert e.value == e.lo == e.hi == 1.0


def test_paired_diff_edges():
    assert paired_diff([], []).value is None
    e = paired_diff([1.0], [0.5])
    assert e.value == 0.5 and e.lo is None
    with pytest.raises(ValueError):
        paired_diff([1.0, 2.0], [1.0])


# ---------------------------------------------------------------------------- mcnemar


def test_mcnemar_values():
    assert mcnemar_p(0, 0) == 1.0
    assert mcnemar_p(3, 3) == 1.0
    assert mcnemar_p(0, 5) == pytest.approx(2 / 32)
    assert mcnemar_p(1, 9) == pytest.approx(2 * 11 / 1024)
    assert mcnemar_p(9, 1) == mcnemar_p(1, 9)
    assert mcnemar_p(0, 2000) == 0.0 or mcnemar_p(0, 2000) < 1e-300


def test_mcnemar_validation():
    with pytest.raises(ValueError):
        mcnemar_p(-1, 3)
    with pytest.raises(ValueError):
        mcnemar_p(1.5, 3)


# ---------------------------------------------------------------------------- brier / ece


def test_brier():
    assert brier([1.0, 0.0], [True, False]) == 0.0
    assert brier([0.8, 0.3], [True, False]) == pytest.approx((0.04 + 0.09) / 2)
    with pytest.raises(InsufficientData):
        brier([], [])


def test_ece_perfect_and_known():
    assert ece([0.25] * 4, [True, False, False, False]) == pytest.approx(0.0)
    # bin [0.9,1.0]: conf 0.95, acc 0.5 ; bin [0.1,0.2): conf 0.15, acc 0
    scores = [0.9, 1.0, 0.15, 0.15]
    labels = [True, False, False, False]
    expected = (2 * abs(0.5 - 0.95) + 2 * abs(0.0 - 0.15)) / 4
    assert ece(scores, labels) == pytest.approx(expected)


def test_ece_last_bin_includes_one():
    assert ece([1.0], [True], bins=10) == 0.0
    assert ece([1.0], [False], bins=10) == 1.0


def test_ece_validation():
    with pytest.raises(InsufficientData):
        ece([], [])
    with pytest.raises(ValueError):
        ece([1.2], [True])
    with pytest.raises(ValueError):
        ece([0.5], [True], bins=0)
    with pytest.raises(ValueError):
        ece([0.5, 0.4], [True])
    with pytest.raises(ValueError):
        ece([float("nan")], [True])


# ---------------------------------------------------------------------------- reliability


def test_reliability_bins():
    scores = [0.05, 0.15, 0.95, 1.0, 0.92]
    labels = [False, False, True, True, False]
    bins = reliability(scores, labels, bins=10)
    assert len(bins) == 10 and all(isinstance(b, Bin) for b in bins)
    assert bins[0].n == 1 and bins[0].accuracy == 0.0 and bins[0].lo == 0.0
    top = bins[9]
    assert top.n == 3 and top.hi == 1.0
    assert top.mean_conf == pytest.approx((0.95 + 1.0 + 0.92) / 3)
    assert top.accuracy == pytest.approx(2 / 3)
    w = wilson(2, 3)
    assert (top.ci_lo, top.ci_hi) == (w.lo, w.hi)
    empty = bins[5]
    assert empty.n == 0 and empty.accuracy is None and (empty.ci_lo, empty.ci_hi) == (0.0, 1.0)
    assert sum(b.n for b in bins) == len(scores)


def test_reliability_validation():
    with pytest.raises(InsufficientData):
        reliability([], [])
    with pytest.raises(ValueError):
        reliability([0.5], ["yes"])


# ---------------------------------------------------------------------------- auroc


def test_auroc_perfect_and_inverted():
    assert auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert auroc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 0.0


def test_auroc_ties_and_known():
    assert auroc([0.5] * 4, [True, False, True, False]) == 0.5
    # pos {0.8, 0.4}, neg {0.6, 0.4}: pairs 0.8>0.6, 0.8>0.4, 0.4<0.6, 0.4=0.4 -> 2.5/4
    assert auroc([0.8, 0.4, 0.6, 0.4], [True, True, False, False]) == pytest.approx(0.625)


def test_auroc_matches_pairwise_definition():
    rng = random.Random(7)
    s = [round(rng.random(), 1) for _ in range(60)]
    y = [rng.random() < 0.4 + 0.4 * v for v in s]
    pos = [a for a, b in zip(s, y, strict=True) if b]
    neg = [a for a, b in zip(s, y, strict=True) if not b]
    brute = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))
    assert auroc(s, y) == pytest.approx(brute)


def test_auroc_one_class_is_none():
    assert auroc([0.1, 0.2], [True, True]) is None
    assert auroc([], []) is None


# ---------------------------------------------------------------------------- risk-coverage


def test_risk_coverage_points():
    s = [0.9, 0.8, 0.8, 0.3]
    y = [True, False, True, False]
    curve = risk_coverage(s, y)
    assert curve == [
        (0.9, 0.25, 0.0),
        (0.8, 0.75, pytest.approx(1 / 3)),
        (0.3, 1.0, 0.5),
    ]
    assert [c for _, c, _ in curve] == sorted(c for _, c, _ in curve)


def test_risk_coverage_empty():
    assert risk_coverage([], []) == []


def test_aurc_perfect_ranking_known_value():
    s = [i / 10 for i in range(10, 0, -1)]
    y = [True] * 7 + [False] * 3
    assert aurc(s, y) == pytest.approx((1 / 8 + 2 / 9 + 3 / 10) / 10)


def test_aurc_perfect_lower_than_random_and_worst():
    rng = random.Random(11)
    n = 200
    y = [rng.random() < 0.7 for _ in range(n)]
    perfect = [0.5 + 0.5 * rng.random() if v else 0.5 * rng.random() for v in y]
    rand = [rng.random() for _ in range(n)]
    worst = [1 - v for v in perfect]
    a_p, a_r, a_w = aurc(perfect, y), aurc(rand, y), aurc(worst, y)
    assert a_p < a_r < a_w
    assert a_r == pytest.approx(0.3, abs=0.08)


def test_aurc_all_tied_equals_overall_risk():
    assert aurc([0.5] * 4, [True, False, False, True]) == pytest.approx(0.5)


def test_aurc_empty_raises():
    with pytest.raises(InsufficientData):
        aurc([], [])


# ---------------------------------------------------------------------------- misc


def test_estimate_is_frozen():
    e = wilson(1, 2)
    with pytest.raises(AttributeError):
        e.value = 0.3  # type: ignore[misc]


def test_public_api_exports():
    for name in (
        "Estimate",
        "Bin",
        "wilson",
        "clopper_pearson",
        "weighted_proportion",
        "mean_ci",
        "bootstrap_ci",
        "paired_diff",
        "mcnemar_p",
        "z_for",
        "required_n",
        "brier",
        "ece",
        "reliability",
        "auroc",
        "risk_coverage",
        "aurc",
    ):
        assert name in stats.__all__


def test_clopper_pearson_matches_binomial_tails():
    # Independent check by direct binomial summation: P[X >= k | lo] = P[X <= k | hi] = alpha/2.
    def tail_ge(k, n, p):
        return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))

    for k, n in [(1, 20), (3, 100), (12, 30), (29, 30)]:
        e = clopper_pearson(k, n)
        assert tail_ge(k, n, e.lo) == pytest.approx(0.025, abs=1e-9)
        assert 1 - tail_ge(k + 1, n, e.hi) == pytest.approx(0.025, abs=1e-9)
