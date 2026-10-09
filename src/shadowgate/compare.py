"""Comparators: decide whether a candidate answer is equivalent to a target answer.

Every comparator implements :class:`shadowgate.types.Comparator` and returns a
:class:`~shadowgate.types.Judgement`. ``equivalent`` is ``None`` only when the comparator could
not decide (unparseable judge reply, backend failure, unparseable multiple-choice target, bad
regex, both sides empty). Shared rules: an empty candidate is never equivalent to a non-empty
target (False); a whitespace-only/empty candidate *and* target is undecidable (None), since
two missing answers do not agree on anything.

Built-in types (see :func:`from_spec`): ``exact``, ``normalized``, ``numeric``, ``choice``,
``contains``, ``regex``, ``judge``.
"""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from shadowgate.errors import ConfigError
from shadowgate.extract import Choice as ChoiceExtractor
from shadowgate.extract import strip_markup
from shadowgate.types import Backend, Comparator, Judgement, Request, Task

__all__ = [
    "Exact",
    "Normalized",
    "Numeric",
    "ChoiceComparator",
    "Contains",
    "RegexComparator",
    "JudgeComparator",
    "ParsedNumber",
    "normalize_text",
    "parse_number",
    "parse_verdict",
    "DEFAULT_JUDGE_PROMPT",
    "from_spec",
]

log = logging.getLogger("shadowgate.compare")


def _empty_mismatch(name: str, candidate: str, target: str) -> Judgement | None:
    if not candidate.strip():
        if target.strip():
            return Judgement(
                equivalent=False, comparator=name, detail={"reason": "empty candidate"}
            )
        return Judgement(equivalent=None, comparator=name, detail={"reason": "both empty"})
    return None


# --------------------------------------------------------------------------- text normalisation

_APOSTROPHES_RE = re.compile(r"['’ʼ`´]")
_ABBREV_DOT_RE = re.compile(r"(?<=[^\W\d_])\.(?=[^\W\d_])")
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_ARTICLES = frozenset({"a", "an", "the"})


def normalize_text(s: str) -> str:
    """Normalise free text for comparison.

    Steps: strip answer markup (bold, backticks, ``\\boxed``), Unicode NFKC, casefold, unicode
    minus -> ``-``, drop thousands separators (``1,000`` -> ``1000``), delete apostrophes and
    dots inside abbreviations (``U.S.`` -> ``us``, ``don't`` -> ``dont``), replace remaining
    punctuation by spaces, collapse whitespace, drop one leading article (a/an/the) when more
    words follow.

    Punctuation that carries numeric meaning is kept: a decimal point or ``/`` between digits
    (``3.5`` != ``35``, ``3/4`` != ``34``), a minus sign directly before a digit at a word
    start (``-3`` != ``3``), ``%``, and ``#`` after a letter (``C#``). Symbols (``$``, ``+``)
    are kept.
    """
    s = strip_markup(s)
    s = unicodedata.normalize("NFKC", s).casefold()
    s = s.replace("\u2212", "-")
    s = _THOUSANDS_RE.sub("", s)
    s = _APOSTROPHES_RE.sub("", s)
    s = _ABBREV_DOT_RE.sub("", s)
    out: list[str] = []
    n = len(s)
    for i, ch in enumerate(s):
        if not unicodedata.category(ch).startswith("P"):
            out.append(ch)
            continue
        prev = s[i - 1] if i > 0 else ""
        nxt = s[i + 1] if i + 1 < n else ""
        if (
            (ch in "./" and prev.isdigit() and nxt.isdigit())
            or (ch == "-" and nxt.isdigit() and (not prev or prev.isspace()))
            or ch == "%"
            or (ch == "#" and prev.isalpha())
        ):
            out.append(ch)
        else:
            out.append(" ")
    s = re.sub(r"\s+%", "%", "".join(out))
    words = s.split()
    if len(words) > 1 and words[0] in _ARTICLES:
        words = words[1:]
    return " ".join(words)


# --------------------------------------------------------------------------- simple comparators


class Exact:
    """Equal after stripping surrounding whitespace (case- and punctuation-sensitive)."""

    name = "exact"

    def compare(self, task: Task, candidate: str, target: str) -> Judgement:
        early = _empty_mismatch(self.name, candidate, target)
        if early:
            return early
        return Judgement(equivalent=candidate.strip() == target.strip(), comparator=self.name)


class Normalized:
    """Equal after :func:`normalize_text`. If both sides normalise to the empty string (e.g. both
    are pure punctuation) the stripped raw strings are compared instead. Two empty answers are
    undecidable (None)."""

    name = "normalized"

    def compare(self, task: Task, candidate: str, target: str) -> Judgement:
        early = _empty_mismatch(self.name, candidate, target)
        if early:
            return early
        a, b = normalize_text(candidate), normalize_text(target)
        eq = candidate.strip() == target.strip() if not a and not b else a == b
        return Judgement(equivalent=eq, comparator=self.name, detail={"candidate": a, "target": b})


class Contains:
    """The normalised target appears in the normalised candidate as a whole-word sequence
    (``"4"`` is not contained in ``"42"``). A target that normalises to nothing is undecidable
    (None), unless the candidate normalises to nothing too, in which case the stripped raw
    strings are compared (as in :class:`Normalized`). Two empty answers are None."""

    name = "contains"

    def compare(self, task: Task, candidate: str, target: str) -> Judgement:
        early = _empty_mismatch(self.name, candidate, target)
        if early:
            return early
        cand, targ = normalize_text(candidate).split(), normalize_text(target).split()
        if not targ:
            if not cand:
                return Judgement(
                    equivalent=candidate.strip() == target.strip(), comparator=self.name
                )
            return Judgement(
                equivalent=None, comparator=self.name, detail={"reason": "empty target"}
            )
        k = len(targ)
        eq = any(cand[i : i + k] == targ for i in range(len(cand) - k + 1))
        return Judgement(equivalent=eq, comparator=self.name)


class RegexComparator:
    """The target is a regular expression the stripped candidate must match.

    ``mode="fullmatch"`` (default) or ``"search"``; ``ignore_case`` defaults to False. An invalid
    target pattern gives ``equivalent=None``.
    """

    name = "regex"

    def __init__(self, mode: str = "fullmatch", ignore_case: bool = False):
        if mode not in ("fullmatch", "search"):
            raise ConfigError("regex comparator: mode must be 'fullmatch' or 'search'")
        self.mode = mode
        self.flags = re.IGNORECASE if ignore_case else 0

    def compare(self, task: Task, candidate: str, target: str) -> Judgement:
        early = _empty_mismatch(self.name, candidate, target)
        if early:
            return early
        try:
            pat = re.compile(target, self.flags)
        except re.error as exc:
            return Judgement(
                equivalent=None, comparator=self.name, detail={"error": f"invalid pattern: {exc}"}
            )
        text = candidate.strip()
        m = pat.fullmatch(text) if self.mode == "fullmatch" else pat.search(text)
        return Judgement(equivalent=m is not None, comparator=self.name)


class ChoiceComparator:
    """Multiple-choice letters: both sides go through the ``choice`` extractor.

    Target without a recognisable letter -> None. Candidate without one -> False (a non-empty
    answer that names no option does not match).
    """

    name = "choice"

    def __init__(self, letters: str = "ABCDEFGHIJ"):
        self._extractor = ChoiceExtractor(letters)

    def compare(self, task: Task, candidate: str, target: str) -> Judgement:
        early = _empty_mismatch(self.name, candidate, target)
        if early:
            return early
        t = self._extractor.extract(target)
        if not t:
            return Judgement(
                equivalent=None, comparator=self.name, detail={"reason": "target has no choice"}
            )
        c = self._extractor.extract(candidate)
        return Judgement(
            equivalent=c == t, comparator=self.name, detail={"candidate": c, "target": t}
        )


# --------------------------------------------------------------------------- numeric

_CURRENCY = {
    "$": "USD", "us$": "USD", "usd": "USD", "dollar": "USD", "dollars": "USD",
    "€": "EUR", "eur": "EUR", "euro": "EUR", "euros": "EUR",
    "£": "GBP", "gbp": "GBP", "pound": "GBP", "pounds": "GBP",
    "¥": "JPY", "jpy": "JPY", "yen": "JPY",
    "₹": "INR", "inr": "INR", "rupee": "INR", "rupees": "INR",
}  # fmt: skip
_NUMBER_FULL_RE = re.compile(
    r"(?:[^\W\d]\w*\s*=\s*)?"
    r"(?:(?:approximately|approx\.?|about|around|roughly|~|≈)\s*)?"
    r"(?P<s1>[-+])?\s*"
    r"(?P<cur>us\$|[$€£¥₹]|usd|eur|gbp|jpy|inr)?\s*"
    r"(?P<s2>[-+])?\s*"
    r"(?P<num>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)"
    r"(?:\s*/\s*(?P<den>\d+(?:\.\d+)?))?"
    r"(?:\s*e(?P<exp>[-+]?\d+)"
    r"|\s*(?:x|×|\*|\\times|\\cdot|·)\s*10\s*\^\s*\{?\s*(?P<exp2>[-+]?\d+)\s*\}?)?"
    r"\s*(?P<pct>%|percent\b|per\s+cent\b)?"
    r"\s*(?P<unit>.*)",
    re.IGNORECASE | re.DOTALL,
)
_UNIT_RE = re.compile(r"[^\W\d_][\w°/^.·\s'’-]*|°\w*", re.UNICODE)
_LATEX_FRAC_RE = re.compile(r"\\[dt]?frac\s*\{\s*([^{}]+?)\s*\}\s*\{\s*([^{}]+?)\s*\}")


@dataclass(frozen=True)
class ParsedNumber:
    """A number read from an answer: exact ``value``, whether it was written as a percentage
    (``value`` is then the number before ``%``), its currency code and trailing unit text.
    ``integer`` is True when it was written as a whole number: digits (optionally with
    thousands separators) without a decimal point, fraction bar or exponent."""

    value: Fraction
    percent: bool = False
    currency: str | None = None
    unit: str = ""
    integer: bool = False


def _canon_unit(unit: str) -> str:
    words = [w for w in re.split(r"\s+", unit.casefold().strip(" .")) if w]
    return " ".join(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words)


def parse_number(text: str) -> ParsedNumber | None:
    """Parse an answer that is a single number, or return None.

    Accepts ints, decimals, thousands separators (``1,234``), fractions (``3/4``,
    ``\\frac{3}{4}``), scientific notation (``1e-3``, ``1.5 x 10^3``), percentages (``45%``,
    ``45 percent``), currency (``$12``, ``-$5``, ``12 USD``, ``12 dollars``), a leading
    ``x =`` or ``approximately``, and a trailing unit of up to three words without digits
    (``12 apples``, ``3.5 km/h``, ``9.8 m/s^2``). Anything containing a second number
    (``12 or 13``) is ambiguous and returns None, as do European decimal commas (``1,5``) and
    mixed numbers (``1 1/2``).
    """
    if not isinstance(text, str):
        return None
    s = unicodedata.normalize("NFKC", text)
    for ch in "\u2212\ufe63\uff0d":
        s = s.replace(ch, "-")
    s = _LATEX_FRAC_RE.sub(r"\1/\2", s)
    s = s.replace("{,}", ",").replace("\\%", "%").replace("\\$", "$").replace("\\,", "")
    s = strip_markup(s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return None
    m = _NUMBER_FULL_RE.fullmatch(s)
    if m is None:
        return None
    s1, s2 = m.group("s1"), m.group("s2")
    if s1 and s2:
        return None
    sign = -1 if (s1 or s2) == "-" else 1
    try:
        value = Fraction(m.group("num").replace(",", "").rstrip(".") or "0")
        if m.group("den") is not None:
            den = Fraction(m.group("den"))
            if den == 0:
                return None
            value /= den
        exp = m.group("exp") or m.group("exp2")
        if exp is not None:
            e = int(exp)
            if abs(e) > 400:
                return None
            value *= Fraction(10) ** e
    except (ValueError, ZeroDivisionError):
        return None
    value *= sign

    currency = _CURRENCY.get(m.group("cur").casefold()) if m.group("cur") else None
    unit = m.group("unit").strip().rstrip(".").strip()
    if unit:
        # exponents such as m/s^2 belong to the unit; any other digit means a second number
        if re.search(r"\d", re.sub(r"\^\s*\{?-?\d+\}?", "", unit)):
            return None  # a second number: ambiguous
        if not _UNIT_RE.fullmatch(unit) or len(unit.split()) > 3:
            return None
        word_cur = _CURRENCY.get(unit.casefold())
        if word_cur:
            if currency and currency != word_cur:
                return None
            currency, unit = word_cur, ""
    num = m.group("num").rstrip(".")  # "42." is a sentence period, still an integer
    integer = (
        "." not in num
        and m.group("den") is None
        and m.group("exp") is None
        and m.group("exp2") is None
    )
    return ParsedNumber(
        value=value,
        percent=m.group("pct") is not None,
        currency=currency,
        unit=unit,
        integer=integer,
    )


_PERCENT_MODES = ("either", "ratio", "strict")


class Numeric:
    """Numeric equivalence. Values are exact rationals and equal values always match. When
    *both* sides are written as integers (``1000000``, ``1,000,001``, ``50%``) only exact
    equality counts, so ``1234567`` != ``1234568`` whatever the tolerances. Otherwise (either
    side is a decimal, fraction or scientific value: ``0.3333333``, ``1/3``, ``1e6``)
    ``rel_tol`` / ``abs_tol`` apply with ``math.isclose`` semantics.

    ``percent`` decides how a percentage compares with a plain number when exactly one side
    has ``%``:

    * ``"either"`` (default): ``50%`` equals both ``0.5`` and ``50``. Datasets disagree on
      whether "what percent" references are stored as ``50`` or ``0.5``; this avoids grading a
      correct answer as wrong for formatting alone.
    * ``"ratio"``: ``50%`` equals ``0.5`` only.
    * ``"strict"``: a percentage only equals another percentage.

    Two percentages always compare their written values. Different currencies on both sides
    or different units on both sides (case and plural ``s`` ignored) are not equivalent; a
    unit or currency on only one side is ignored (``12 apples`` == ``12``).

    When either side is not a single number: ``fallback_text=True`` (default) compares with
    :func:`normalize_text`, else ``equivalent=None``.
    """

    name = "numeric"

    def __init__(
        self,
        rel_tol: float = 1e-6,
        abs_tol: float = 1e-9,
        fallback_text: bool = True,
        percent: str = "either",
    ):
        if percent not in _PERCENT_MODES:
            raise ConfigError(f"numeric comparator: percent must be one of {_PERCENT_MODES}")
        if rel_tol < 0 or abs_tol < 0:
            raise ConfigError("numeric comparator: tolerances must be >= 0")
        self.rel_tol = float(rel_tol)
        self.abs_tol = float(abs_tol)
        self.fallback_text = bool(fallback_text)
        self.percent = percent

    def _close(self, a: Fraction, b: Fraction, *, exact: bool = False) -> bool:
        if a == b:
            return True
        if exact:
            return False
        try:
            return math.isclose(float(a), float(b), rel_tol=self.rel_tol, abs_tol=self.abs_tol)
        except OverflowError:
            return False

    def compare(self, task: Task, candidate: str, target: str) -> Judgement:
        early = _empty_mismatch(self.name, candidate, target)
        if early:
            return early
        c, t = parse_number(candidate), parse_number(target)
        if c is None or t is None:
            side = "candidate" if c is None else "target"
            if not self.fallback_text:
                return Judgement(
                    equivalent=None,
                    comparator=self.name,
                    detail={"reason": f"{side} is not numeric"},
                )
            a, b = normalize_text(candidate), normalize_text(target)
            return Judgement(
                equivalent=a == b,
                comparator=self.name,
                detail={"reason": f"{side} is not numeric", "fallback": "normalized"},
            )
        detail: dict[str, Any] = {"candidate": _fmt(c), "target": _fmt(t)}
        if c.currency and t.currency and c.currency != t.currency:
            detail["reason"] = "currency mismatch"
            return Judgement(equivalent=False, comparator=self.name, detail=detail)
        if c.unit and t.unit and _canon_unit(c.unit) != _canon_unit(t.unit):
            detail["reason"] = "unit mismatch"
            return Judgement(equivalent=False, comparator=self.name, detail=detail)
        exact = c.integer and t.integer
        if exact:
            detail["exact"] = True
        if c.percent == t.percent:
            eq = self._close(c.value, t.value, exact=exact)
        else:
            pct, plain = (c, t) if c.percent else (t, c)
            as_ratio = self._close(pct.value / 100, plain.value, exact=exact)
            if self.percent == "strict":
                eq = False
            elif self.percent == "ratio":
                eq = as_ratio
            else:
                eq = as_ratio or self._close(pct.value, plain.value, exact=exact)
            detail["percent"] = self.percent
        return Judgement(equivalent=eq, comparator=self.name, detail=detail)


def _fmt(p: ParsedNumber) -> str:
    v = p.value
    s = str(v.numerator) if v.denominator == 1 else repr(float(v))
    return s + ("%" if p.percent else "")


# --------------------------------------------------------------------------- judge

DEFAULT_JUDGE_PROMPT = """\
Decide whether a candidate answer is equivalent to a reference answer for the question below.
Answers are EQUIVALENT when they state the same final result, even if formatting, wording,
units notation or the amount of explanation differ. They are DIFFERENT when the final results
differ, or when the candidate is missing, hedges between several results, or is ambiguous.

Question:
{question}

Reference answer:
{target}

Candidate answer:
{candidate}

Give a one-sentence justification, then end with a final line that is exactly
VERDICT: EQUIVALENT
or
VERDICT: DIFFERENT
"""

_VERDICT_RE = re.compile(
    r"verdict\s*[*_]*\s*[:=\-]?\s*[*_`\"'\[]*\s*"
    r"(not[\s_-]+equivalent|non[\s_-]?equivalent|equivalent|different)\b",
    re.IGNORECASE,
)
_PLACEHOLDER_RE = re.compile(r"\{(question|candidate|target)\}")


def parse_verdict(text: str) -> bool | None:
    """Last ``VERDICT: EQUIVALENT|DIFFERENT`` in ``text`` (case-insensitive; markdown
    decoration and ``NOT EQUIVALENT`` tolerated) -> True / False; None if absent."""
    found = _VERDICT_RE.findall(text or "")
    if not found:
        return None
    word = found[-1].casefold()
    return word == "equivalent"


class JudgeComparator:
    """Ask a model whether candidate and target are equivalent.

    ``prompt`` must contain ``{candidate}`` and ``{target}`` and may contain ``{question}``
    (the task prompt). Only these three placeholders are substituted; other braces are left
    as written. The request carries ``tags={"role": "judge", "task_id": ...}``. The judge's
    completion is recorded in ``Judgement.calls``. A backend failure or a reply without a
    verdict gives ``equivalent=None`` with the reason in ``detail``.

    With ``shortcut=True`` (default), answers that are already equal after
    :func:`normalize_text` are judged equivalent without a model call.
    """

    name = "judge"

    def __init__(
        self,
        backend: Backend,
        prompt: str = DEFAULT_JUDGE_PROMPT,
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float | None = None,
        effort: str | None = None,
        shortcut: bool = True,
    ):
        if not isinstance(prompt, str) or "{candidate}" not in prompt or "{target}" not in prompt:
            raise ConfigError("judge prompt must contain {candidate} and {target}")
        if not hasattr(backend, "complete"):
            raise ConfigError("judge backend must implement complete(request)")
        self.backend = backend
        self.prompt = prompt
        self.system = system
        self.max_tokens = int(max_tokens)
        self.temperature = temperature
        self.effort = effort
        self.shortcut = bool(shortcut)

    def render(self, question: str, candidate: str, target: str) -> str:
        values = {"question": question, "candidate": candidate, "target": target}
        return _PLACEHOLDER_RE.sub(lambda m: values[m.group(1)], self.prompt)

    def compare(self, task: Task, candidate: str, target: str) -> Judgement:
        early = _empty_mismatch(self.name, candidate, target)
        if early:
            return early
        if self.shortcut and normalize_text(candidate) == normalize_text(target):
            return Judgement(
                equivalent=True, comparator=self.name, detail={"reason": "normalized match"}
            )
        request = Request(
            prompt=self.render(task.prompt, candidate.strip(), target.strip()),
            system=self.system,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            effort=self.effort,
            tags={"role": "judge", "task_id": task.id},
        )
        try:
            completion = self.backend.complete(request)
        except Exception as exc:  # judged as undecidable, never as agree/disagree
            log.warning("judge backend %s failed: %s", getattr(self.backend, "name", "?"), exc)
            return Judgement(
                equivalent=None,
                comparator=self.name,
                detail={"error": f"{type(exc).__name__}: {exc}"},
            )
        verdict = parse_verdict(completion.text)
        detail: dict[str, Any] = {"backend": getattr(self.backend, "name", "")}
        if verdict is None:
            detail["error"] = "unparseable judge reply"
        else:
            detail["verdict"] = "EQUIVALENT" if verdict else "DIFFERENT"
        return Judgement(
            equivalent=verdict, comparator=self.name, calls=(completion,), detail=detail
        )


# --------------------------------------------------------------------------- factory

_SIMPLE: dict[str, tuple[type, frozenset[str]]] = {
    "exact": (Exact, frozenset()),
    "normalized": (Normalized, frozenset()),
    "numeric": (Numeric, frozenset({"rel_tol", "abs_tol", "fallback_text", "percent"})),
    "choice": (ChoiceComparator, frozenset({"letters"})),
    "contains": (Contains, frozenset()),
    "regex": (RegexComparator, frozenset({"mode", "ignore_case"})),
}
_JUDGE_KEYS = frozenset(
    {"backend", "prompt", "system", "max_tokens", "temperature", "effort", "shortcut"}
)


def from_spec(
    spec: Mapping[str, Any] | str | None, *, backends: Mapping[str, Backend] | None = None
) -> Comparator:
    """Build a comparator from ``{"type": ..., **options}``; ``None`` -> ``Normalized()``.

    A bare string is shorthand for ``{"type": <string>}``. ``judge`` needs ``backend``: a name
    looked up in ``backends``. Unknown types, unknown keys or a missing backend raise
    :class:`~shadowgate.errors.ConfigError`.
    """
    if spec is None:
        return Normalized()
    if isinstance(spec, str):
        spec = {"type": spec}
    if not isinstance(spec, Mapping):
        raise ConfigError(f"comparator spec must be a table, got {type(spec).__name__}")
    kind = spec.get("type")
    opts = {k: v for k, v in spec.items() if k != "type"}
    if kind == "judge":
        for key in opts:
            if key not in _JUDGE_KEYS:
                raise ConfigError(f"unknown key {key!r} for comparator type 'judge'")
        name = opts.pop("backend", None)
        if not name:
            raise ConfigError("comparator type 'judge' requires key 'backend'")
        backends = backends or {}
        if name not in backends:
            known = ", ".join(sorted(backends)) or "none"
            raise ConfigError(f"judge backend {name!r} is not defined (known: {known})")
        try:
            return JudgeComparator(backends[name], **opts)
        except TypeError as exc:
            raise ConfigError(f"invalid options for comparator type 'judge': {exc}") from exc
    if kind not in _SIMPLE:
        known = ", ".join(sorted([*_SIMPLE, "judge"]))
        raise ConfigError(f"unknown comparator type {kind!r}; expected one of {known}")
    cls, allowed = _SIMPLE[kind]
    for key in opts:
        if key not in allowed:
            raise ConfigError(f"unknown key {key!r} for comparator type {kind!r}")
    try:
        return cls(**opts)
    except TypeError as exc:
        raise ConfigError(f"invalid options for comparator type {kind!r}: {exc}") from exc
