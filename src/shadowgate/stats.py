"""Interval estimators, calibration metrics and resampling helpers (stdlib only).

Everything here is pure Python (``math`` + ``random``) so the package has no runtime
dependencies. Functions validate their input and raise :class:`ValueError` for malformed input
(wrong lengths, values out of range, NaN) and :class:`~shadowgate.errors.InsufficientData` when a
statistic cannot be computed from too few observations.

Empty-input conventions
-----------------------
* Proportion estimators (:func:`wilson`, :func:`clopper_pearson`,
  :func:`weighted_proportion`) with zero observations return an :class:`Estimate` with
  ``value=None`` and the vacuous interval ``lo=0.0, hi=1.0`` (``n=0``): nothing is known, but the
  rate is still certainly in ``[0, 1]``.
* Mean-type estimators (:func:`mean_ci`, :func:`paired_diff`) with zero observations return
  ``value=None, lo=None, hi=None``; with a single observation they return the point value and
  ``lo=hi=None`` (no spread can be estimated).
* :func:`bootstrap_ci`, :func:`brier`, :func:`ece`, :func:`reliability` and :func:`aurc` raise
  :class:`InsufficientData` on empty input.
* :func:`auroc` returns ``None`` when either class is absent (including empty input);
  :func:`risk_coverage` returns ``[]`` on empty input; :func:`mcnemar_p` returns ``1.0`` when there
  are no discordant pairs.

References
----------
* Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference.
* Clopper, C. J. & Pearson, E. S. (1934). The use of confidence or fiducial limits illustrated in
  the case of the binomial.
* Hajek, J. (1971); Sarndal, Swensson & Wretman (1992), *Model Assisted Survey Sampling*,
  ch. 5.7 (ratio estimator and its Taylor-linearized variance).
* Kish, L. (1965). *Survey Sampling* (effective sample size under unequal weighting).
* Korn, E. L. & Graubard, B. I. (1998). Confidence intervals for proportions with small expected
  number of positive counts estimated from survey data. *Survey Methodology* 24(2), 193-201.
* Acklam, P. J. (2003). An algorithm for computing the inverse normal cumulative distribution
  function.
* Press et al., *Numerical Recipes* (3rd ed.), 6.4 (incomplete beta by continued fraction,
  modified Lentz's method).
* Efron, B. & Tibshirani, R. (1993). *An Introduction to the Bootstrap* (percentile interval).
* McNemar, Q. (1947); exact conditional (binomial) version of the test.
* Naeini, Cooper & Hauskrecht (2015) (expected calibration error); Geifman & El-Yaniv (2017)
  and Geifman et al. (2019) (risk-coverage curve, AURC).
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from shadowgate.errors import InsufficientData

__all__ = [
    "Bin",
    "Estimate",
    "aurc",
    "auroc",
    "bootstrap_ci",
    "brier",
    "clopper_pearson",
    "ece",
    "mcnemar_p",
    "mean_ci",
    "paired_diff",
    "regularized_beta",
    "reliability",
    "required_n",
    "risk_coverage",
    "weighted_bounds",
    "weighted_proportion",
    "wilson",
    "z_for",
]


# --------------------------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Estimate:
    """A point estimate with a two-sided confidence interval.

    ``value``/``lo``/``hi`` may be ``None`` when the data cannot support them (see the module
    docstring for the empty-input conventions). ``n`` is the number of observations, ``method`` a
    short identifier of the estimator, ``level`` the nominal confidence level and ``n_eff`` the
    effective sample size used for the interval (weighted estimators only).
    """

    value: float | None
    lo: float | None
    hi: float | None
    n: int
    method: str
    level: float = 0.95
    n_eff: float | None = None


@dataclass(frozen=True)
class Bin:
    """One bin of a reliability diagram.

    ``mean_conf`` and ``accuracy`` are ``None`` for an empty bin; ``ci_lo``/``ci_hi`` is the Wilson
    interval for the bin accuracy (``0.0``/``1.0`` for an empty bin).
    """

    lo: float
    hi: float
    n: int
    mean_conf: float | None
    accuracy: float | None
    ci_lo: float
    ci_hi: float


# --------------------------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------------------------


def _check_level(level: float) -> float:
    if isinstance(level, bool) or not isinstance(level, (int, float)):
        raise ValueError(f"level must be a number in (0, 1), got {level!r}")
    level = float(level)
    if not (0.0 < level < 1.0):
        raise ValueError(f"level must be in (0, 1), got {level!r}")
    return level


def _check_count(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def _check_kn(k: int, n: int) -> None:
    _check_count("k", k)
    _check_count("n", n)
    if k > n:
        raise ValueError(f"k must be <= n, got k={k}, n={n}")


def _as_label(x: object) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, int) and x in (0, 1):
        return bool(x)
    raise ValueError(f"labels must be bool (or 0/1), got {x!r}")


def _as_finite(name: str, x: object) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"{name} must be numeric, got {x!r}")
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {x!r}")
    return v


def _scores_labels(
    scores: Sequence[float], labels: Sequence[bool]
) -> tuple[list[float], list[bool]]:
    s = list(scores)
    y = list(labels)
    if len(s) != len(y):
        raise ValueError(f"scores and labels differ in length ({len(s)} vs {len(y)})")
    out_s: list[float] = []
    for v in s:
        f = _as_finite("score", v)
        if not (0.0 <= f <= 1.0):
            raise ValueError(f"scores must be in [0, 1], got {v!r}")
        out_s.append(f)
    return out_s, [_as_label(v) for v in y]


# --------------------------------------------------------------------------------------------
# Normal quantile
# --------------------------------------------------------------------------------------------

_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)
_P_LOW = 0.02425


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF, Phi^-1(p), for 0 < p < 1.

    Acklam's rational approximation (relative error < 1.15e-9) followed by one Halley step
    using ``math.erfc``:  e = Phi(x) - p,  u = e * sqrt(2 pi) * exp(x^2 / 2),
    x <- x - u / (1 + x u / 2). The result is accurate to near machine precision.
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be in (0, 1), got {p!r}")
    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        )
    elif p <= 1.0 - _P_LOW:
        q = p - 0.5
        r = q * q
        x = (
            (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5])
            * q
            / (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0)
        )
    else:
        q = math.sqrt(-2.0 * math.log1p(-p))
        x = -(
            (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5])
            / ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
        )
    e = 0.5 * math.erfc(-x / math.sqrt(2.0)) - p
    u = e * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
    return x - u / (1.0 + x * u / 2.0)


def z_for(level: float) -> float:
    """Two-sided normal critical value: z = Phi^-1(1 - (1 - level) / 2).

    Computed as ``-Phi^-1((1 - level) / 2)`` so the small tail probability is used directly
    (better relative accuracy). E.g. ``z_for(0.95) = 1.959963984540054``.
    """
    level = _check_level(level)
    return -_norm_ppf((1.0 - level) / 2.0)


# --------------------------------------------------------------------------------------------
# Binomial intervals
# --------------------------------------------------------------------------------------------


def _wilson_bounds(p: float, n: float, z: float) -> tuple[float, float]:
    """Wilson score bounds for proportion ``p`` observed on (possibly fractional) size ``n``.

    centre = (p + z^2/(2n)) / (1 + z^2/n)
    half   = z / (1 + z^2/n) * sqrt(p(1-p)/n + z^2/(4n^2))
    Bounds are clamped to [0, 1] and set exactly to 0 (1) when p == 0 (p == 1).
    """
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = z / denom * math.sqrt(max(p * (1.0 - p) / n + z2 / (4.0 * n * n), 0.0))
    lo = 0.0 if p <= 0.0 else max(0.0, centre - half)
    hi = 1.0 if p >= 1.0 else min(1.0, centre + half)
    return lo, hi


def wilson(k: int, n: int, level: float = 0.95) -> Estimate:
    """Wilson score interval for a binomial proportion k/n.

    With p = k/n and z = z_for(level), the bounds are
    (p + z^2/2n -/+ z sqrt(p(1-p)/n + z^2/4n^2)) / (1 + z^2/n).
    ``n == 0`` returns ``Estimate(value=None, lo=0.0, hi=1.0, n=0)``.
    """
    level = _check_level(level)
    _check_kn(k, n)
    if n == 0:
        return Estimate(None, 0.0, 1.0, 0, "wilson", level)
    p = k / n
    lo, hi = _wilson_bounds(p, float(n), z_for(level))
    return Estimate(p, lo, hi, n, "wilson", level)


def regularized_beta(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta function I_x(a, b) for a, b > 0, 0 <= x <= 1.

    I_x(a,b) = x^a (1-x)^b / (a B(a,b)) * CF(x; a, b), with the continued fraction evaluated by
    the modified Lentz method (Numerical Recipes 6.4). For x >= (a+1)/(a+b+2) the symmetry
    I_x(a,b) = 1 - I_{1-x}(b,a) is used so the fraction converges quickly.
    """
    if not (a > 0 and b > 0):
        raise ValueError(f"a and b must be > 0, got a={a!r}, b={b!r}")
    if not (0.0 <= x <= 1.0):
        raise ValueError(f"x must be in [0, 1], got {x!r}")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(log_front) * _betacf(x, a, b) / a
    return 1.0 - math.exp(log_front) * _betacf(1.0 - x, b, a) / b


def _betacf(x: float, a: float, b: float) -> float:
    tiny = 1e-300
    eps = 1e-16
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 10_000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    return h  # pragma: no cover - convergence is reached far earlier for valid inputs


def _beta_ppf(q: float, a: float, b: float) -> float:
    """Quantile of Beta(a, b) by bisection on the (monotone) regularized incomplete beta."""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if regularized_beta(mid, a, b) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-15:
            break
    return 0.5 * (lo + hi)


def _cp_bounds(x: float, n: float, level: float) -> tuple[float, float]:
    """Clopper-Pearson bounds for ``x`` successes in ``n`` trials (both may be fractional).

    lo = Beta^-1(alpha/2; x, n-x+1) (0 when x <= 0), hi = Beta^-1(1-alpha/2; x+1, n-x)
    (1 when x >= n), using the closed forms (alpha/2)^(1/n) at x = n and 1 - (alpha/2)^(1/n) at
    x = 0. The beta quantiles are found by bisection on :func:`regularized_beta`, which accepts
    non-integer parameters, so the same formula serves the Korn-Graubard interval.
    """
    alpha = 1.0 - level
    if x <= 0.0:
        return 0.0, 1.0 - (alpha / 2.0) ** (1.0 / n)
    if x >= n:
        return (alpha / 2.0) ** (1.0 / n), 1.0
    return _beta_ppf(alpha / 2.0, x, n - x + 1.0), _beta_ppf(1.0 - alpha / 2.0, x + 1.0, n - x)


def clopper_pearson(k: int, n: int, level: float = 0.95) -> Estimate:
    """Exact (Clopper-Pearson) binomial interval.

    lo = Beta^-1(alpha/2; k, n-k+1)   (0 when k = 0)
    hi = Beta^-1(1-alpha/2; k+1, n-k) (1 when k = n)
    with alpha = 1 - level. The k = 0 / k = n bounds use the closed forms
    hi = 1 - (alpha/2)^(1/n) and lo = (alpha/2)^(1/n). ``n == 0`` returns the vacuous interval.
    """
    level = _check_level(level)
    _check_kn(k, n)
    if n == 0:
        return Estimate(None, 0.0, 1.0, 0, "clopper-pearson", level)
    lo, hi = _cp_bounds(float(k), float(n), level)
    return Estimate(k / n, lo, hi, n, "clopper-pearson", level)


WEIGHTED_METHODS = ("korn-graubard", "wilson")


def weighted_proportion(
    ys: Sequence[bool],
    weights: Sequence[float],
    level: float = 0.95,
    method: str = "korn-graubard",
) -> Estimate:
    """Hajek (ratio) estimate of a population proportion from an unequal-probability sample.

    With inclusion probabilities pi_i and weights w_i = 1/pi_i:

    * point estimate   p = sum(w y) / sum(w)
    * linearized var   v = sum(w^2 (y - p)^2) / (sum w)^2   (Taylor linearization of the ratio,
      with-replacement approximation: ignores the finite-population correction, so it is mildly
      conservative for Poisson/Bernoulli sampling without replacement)
    * Kish effective n n_kish = (sum w)^2 / sum(w^2)
    * variance-matched effective n n_lin = p(1-p)/v  (when 0 < p < 1 and v > 0)

    The effective size is n_eff = min(n_kish, n_lin) (n_kish alone when p is 0 or 1): the
    smaller of the two accounts both for the dispersion of the weights (Kish) and for errors
    concentrated in heavily weighted units (linearization), which Kish's formula ignores.
    ``Estimate.n_eff`` reports it.

    ``method`` picks the interval evaluated at (p, n_eff):

    * ``"korn-graubard"`` (default, ``Estimate.method == "hajek-korn-graubard"``): the
      Clopper-Pearson interval with the non-integer count x_eff = p * n_eff out of n_eff
      (Korn & Graubard 1998, without their degrees-of-freedom adjustment), computed with the
      regularized incomplete beta. It is conservative where Wilson under-covers: with a few
      heavily weighted audits in a low-pi stratum the Hajek estimate is skewed and the Wilson
      interval misses on the high side (simulated coverage 0.87 at 95%).
    * ``"wilson"`` (``"hajek-wilson"``): the Wilson score interval at (p, n_eff); narrower.

    With equal weights n_kish = n_lin = n, so the result coincides with
    ``clopper_pearson(sum(y), n)`` (``wilson(sum(y), n)`` for method "wilson").

    Weights must be finite and > 0. Empty input returns the vacuous interval.
    """
    level = _check_level(level)
    if method not in WEIGHTED_METHODS:
        raise ValueError(f"method must be one of {WEIGHTED_METHODS}, got {method!r}")
    name = f"hajek-{method}"
    y = [_as_label(v) for v in ys]
    w = [_as_finite("weight", v) for v in weights]
    if len(y) != len(w):
        raise ValueError(f"ys and weights differ in length ({len(y)} vs {len(w)})")
    if any(v <= 0.0 for v in w):
        raise ValueError("weights must be > 0")
    n = len(y)
    if n == 0:
        return Estimate(None, 0.0, 1.0, 0, name, level, n_eff=0.0)
    k = sum(y)
    if all(v == w[0] for v in w):
        # Equal weights: the Hajek estimator is exactly the sample proportion.
        p, n_eff = k / n, float(n)
    else:
        sw = math.fsum(w)
        sw2 = math.fsum(v * v for v in w)
        p = math.fsum(wi for wi, yi in zip(w, y, strict=True) if yi) / sw
        p = min(1.0, max(0.0, p))
        n_eff = sw * sw / sw2
        if 0.0 < p < 1.0:
            var = math.fsum(
                wi * wi * ((1.0 if yi else 0.0) - p) ** 2 for wi, yi in zip(w, y, strict=True)
            ) / (sw * sw)
            if var > 0.0:
                n_eff = min(n_eff, p * (1.0 - p) / var)
    lo, hi = weighted_bounds(p, n_eff, level, method)
    return Estimate(p, lo, hi, n, name, level, n_eff=n_eff)


def weighted_bounds(p: float, n_eff: float, level: float, method: str) -> tuple[float, float]:
    """Interval bounds of :func:`weighted_proportion` at proportion ``p`` and effective size
    ``n_eff`` (> 0); ``method`` is "korn-graubard" or "wilson". Used for sample-size planning."""
    if method == "wilson":
        return _wilson_bounds(p, n_eff, z_for(level))
    return _cp_bounds(p * n_eff, n_eff, level)


def required_n(p: float, half_width: float, level: float = 0.95) -> int:
    """Sample size for a target CI half-width (Wald approximation).

    n = ceil(z^2 p (1 - p) / h^2), with z = z_for(level). This is the usual planning formula; the
    Wilson interval actually reported is close to it for moderate p and n. Returns at least 1.
    Requires 0 <= p <= 1 and 0 < half_width < 1.
    """
    level = _check_level(level)
    p = _as_finite("p", p)
    h = _as_finite("half_width", half_width)
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0, 1], got {p!r}")
    if not (0.0 < h < 1.0):
        raise ValueError(f"half_width must be in (0, 1), got {half_width!r}")
    z = z_for(level)
    raw = z * z * p * (1.0 - p) / (h * h)
    return max(1, math.ceil(raw - 1e-9))


# --------------------------------------------------------------------------------------------
# Means, bootstrap, paired comparisons
# --------------------------------------------------------------------------------------------


def mean_ci(xs: Sequence[float], level: float = 0.95) -> Estimate:
    """Mean with a normal-approximation interval: mean -/+ z * s / sqrt(n).

    s is the sample standard deviation (n - 1 denominator). No t correction is applied, so the
    interval is slightly narrow for very small n. Empty input gives ``value=None, lo=hi=None``; a
    single observation gives the value with ``lo=hi=None``.
    """
    level = _check_level(level)
    x = [_as_finite("x", v) for v in xs]
    n = len(x)
    if n == 0:
        return Estimate(None, None, None, 0, "normal", level)
    m = math.fsum(x) / n
    if n == 1:
        return Estimate(m, None, None, 1, "normal", level)
    s2 = math.fsum((v - m) ** 2 for v in x) / (n - 1)
    half = z_for(level) * math.sqrt(s2 / n)
    return Estimate(m, m - half, m + half, n, "normal", level)


def _quantile(sorted_vals: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile (Hyndman & Fan type 7) of pre-sorted values."""
    pos = q * (len(sorted_vals) - 1)
    i = math.floor(pos)
    j = min(i + 1, len(sorted_vals) - 1)
    frac = pos - i
    return sorted_vals[i] + (sorted_vals[j] - sorted_vals[i]) * frac


def bootstrap_ci(
    stat: Callable[[Sequence[int]], float],
    n: int,
    *,
    reps: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for a statistic of ``n`` observations.

    Each replicate draws ``n`` indices uniformly with replacement (``random.Random(seed)``) and
    calls ``stat(indices)``; the interval is the (alpha/2, 1 - alpha/2) empirical quantiles of the
    replicate values (linear interpolation). Deterministic for a given seed.
    """
    level = _check_level(level)
    _check_count("n", n)
    _check_count("reps", reps)
    if n == 0:
        raise InsufficientData("bootstrap_ci needs at least one observation")
    if reps < 1:
        raise ValueError("reps must be >= 1")
    rng = random.Random(seed)
    population = range(n)
    vals = []
    for _ in range(reps):
        v = float(stat(rng.choices(population, k=n)))
        if math.isnan(v):
            raise ValueError("stat returned NaN")
        vals.append(v)
    vals.sort()
    alpha = 1.0 - level
    return _quantile(vals, alpha / 2.0), _quantile(vals, 1.0 - alpha / 2.0)


def paired_diff(
    a: Sequence[float], b: Sequence[float], level: float = 0.95, reps: int = 2000, seed: int = 0
) -> Estimate:
    """Mean paired difference mean(a_i - b_i) with a percentile-bootstrap interval.

    Pairs are resampled jointly (the bootstrap is over the differences d_i = a_i - b_i), which
    preserves the within-pair correlation. Empty input gives ``value=None``; a single pair gives
    the value with ``lo=hi=None``.
    """
    level = _check_level(level)
    xa = [_as_finite("a", v) for v in a]
    xb = [_as_finite("b", v) for v in b]
    if len(xa) != len(xb):
        raise ValueError(f"a and b differ in length ({len(xa)} vs {len(xb)})")
    d = [u - v for u, v in zip(xa, xb, strict=True)]
    n = len(d)
    if n == 0:
        return Estimate(None, None, None, 0, "paired-bootstrap", level)
    m = math.fsum(d) / n
    if n == 1:
        return Estimate(m, None, None, 1, "paired-bootstrap", level)
    lo, hi = bootstrap_ci(
        lambda idx: math.fsum(d[i] for i in idx) / len(idx), n, reps=reps, level=level, seed=seed
    )
    return Estimate(m, lo, hi, n, "paired-bootstrap", level)


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar test on discordant counts b and c.

    Under H0 the discordant pairs split as Binomial(b + c, 1/2), so
    p = min(1, 2 * P[X <= min(b, c)]),  X ~ Bin(b + c, 1/2), computed in exact integer
    arithmetic. Returns 1.0 when b + c == 0.
    """
    _check_count("b", b)
    _check_count("c", c)
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return min(1.0, 2 * tail / 2**n)


# --------------------------------------------------------------------------------------------
# Calibration and selective prediction
# --------------------------------------------------------------------------------------------


def brier(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Brier score: mean (s_i - y_i)^2 with y_i = 1 for a correct (True) label."""
    s, y = _scores_labels(scores, labels)
    if not s:
        raise InsufficientData("brier needs at least one observation")
    return math.fsum((si - (1.0 if yi else 0.0)) ** 2 for si, yi in zip(s, y, strict=True)) / len(s)


def _bin_index(s: float, bins: int) -> int:
    # Equal-width bins [i/B, (i+1)/B); the last bin also includes 1.0.
    return min(int(s * bins), bins - 1)


def _check_bins(bins: int) -> int:
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError(f"bins must be a positive int, got {bins!r}")
    return bins


def ece(scores: Sequence[float], labels: Sequence[bool], bins: int = 10) -> float:
    """Expected calibration error with equal-width bins.

    ECE = sum_b (n_b / N) * |acc_b - conf_b|, bins [i/B, (i+1)/B) with the last bin closed so a
    score of 1.0 is counted; empty bins contribute nothing.
    """
    _check_bins(bins)
    s, y = _scores_labels(scores, labels)
    if not s:
        raise InsufficientData("ece needs at least one observation")
    total = 0.0
    for b in reliability(s, y, bins):
        if b.n:
            assert b.accuracy is not None and b.mean_conf is not None
            total += b.n * abs(b.accuracy - b.mean_conf)
    return total / len(s)


def reliability(
    scores: Sequence[float], labels: Sequence[bool], bins: int = 10, level: float = 0.95
) -> list[Bin]:
    """Reliability-diagram bins (always ``bins`` entries, equal width, last bin includes 1.0).

    Each :class:`Bin` carries the count, mean confidence, observed accuracy and the Wilson
    interval of the accuracy at ``level``. Empty bins have ``n=0``, ``None`` statistics and the
    interval [0, 1].
    """
    _check_bins(bins)
    level = _check_level(level)
    s, y = _scores_labels(scores, labels)
    if not s:
        raise InsufficientData("reliability needs at least one observation")
    sums = [0.0] * bins
    hits = [0] * bins
    counts = [0] * bins
    for si, yi in zip(s, y, strict=True):
        i = _bin_index(si, bins)
        sums[i] += si
        hits[i] += int(yi)
        counts[i] += 1
    out: list[Bin] = []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        if counts[i] == 0:
            out.append(Bin(lo, hi, 0, None, None, 0.0, 1.0))
            continue
        w = wilson(hits[i], counts[i], level)
        assert w.lo is not None and w.hi is not None
        out.append(Bin(lo, hi, counts[i], sums[i] / counts[i], hits[i] / counts[i], w.lo, w.hi))
    return out


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    """Area under the ROC curve for separating True from False labels by score.

    Mann-Whitney form: AUC = (R_pos - n_pos (n_pos + 1) / 2) / (n_pos n_neg), where R_pos is the
    rank sum of the positives using average ranks for ties (a tie counts as 1/2). Returns ``None``
    when either class is absent.
    """
    s, y = _scores_labels(scores, labels)
    n_pos = sum(y)
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(range(len(s)), key=lambda i: s[i])
    ranks = [0.0] * len(s)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and s[order[j + 1]] == s[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    r_pos = math.fsum(r for r, yi in zip(ranks, y, strict=True) if yi)
    return (r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def risk_coverage(
    scores: Sequence[float], labels: Sequence[bool]
) -> list[tuple[float, float, float]]:
    """Risk-coverage curve of the selective rule "accept when score >= t".

    One point per distinct score t (taken in descending order): coverage(t) = share of items with
    score >= t, risk(t) = share of label-False items among those accepted. Returned as
    ``(threshold, coverage, risk)`` sorted by increasing coverage; the last point has coverage 1.
    Empty input returns ``[]``.
    """
    s, y = _scores_labels(scores, labels)
    n = len(s)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: -s[i])
    out: list[tuple[float, float, float]] = []
    errors = 0
    i = 0
    while i < n:
        t = s[order[i]]
        while i < n and s[order[i]] == t:
            errors += 0 if y[order[i]] else 1
            i += 1
        out.append((t, i / n, errors / i))
    return out


def aurc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Area under the risk-coverage curve (lower is better).

    Step-function integral over coverage from 0 to 1: AURC = sum_j risk_j * (cov_j - cov_{j-1})
    with cov_0 = 0, over the points of :func:`risk_coverage`. Equivalently, the mean over the N
    unit coverage steps of the risk at that step, where tied scores share the risk of their tie
    group; without ties this is the AURC of Geifman et al. (2019).
    """
    curve = risk_coverage(scores, labels)
    if not curve:
        raise InsufficientData("aurc needs at least one observation")
    total = 0.0
    prev = 0.0
    for _, cov, risk in curve:
        total += risk * (cov - prev)
        prev = cov
    return total
