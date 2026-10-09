from __future__ import annotations

from fractions import Fraction

import pytest

from shadowgate.compare import (
    DEFAULT_JUDGE_PROMPT,
    ChoiceComparator,
    Contains,
    Exact,
    JudgeComparator,
    Normalized,
    Numeric,
    RegexComparator,
    from_spec,
    normalize_text,
    parse_number,
    parse_verdict,
)
from shadowgate.errors import BackendError, ConfigError
from shadowgate.types import Comparator, Completion, Request, Task

TASK = Task(id="t1", prompt="What is the capital of France?")


class FakeBackend:
    """Minimal ``types.Backend``: returns canned replies and records requests."""

    def __init__(self, reply: str = "", *, error: Exception | None = None) -> None:
        self.name = "fake:judge"
        self.reply = reply
        self.error = error
        self.requests: list[Request] = []

    def complete(self, request: Request) -> Completion:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return Completion(text=self.reply, model="fake", cost_usd=0.001)


# --------------------------------------------------------------------------- normalize_text

NORMALIZE_CASES = [
    ("The Eiffel Tower!", "eiffel tower"),
    ("  Paris.  ", "paris"),
    ("U.S.A.", "usa"),
    ("don't", "dont"),
    ("Hello,   world", "hello world"),
    ("ＡＢＣ", "abc"),
    ("Straße", "strasse"),
    ("An apple", "apple"),
    ("a", "a"),
    ("the", "the"),
    ("The The", "the"),
    ("-3", "-3"),
    ("−3", "-3"),
    ("3.5", "3.5"),
    ("1,000", "1000"),
    ("3/4", "3/4"),
    ("50 %", "50%"),
    ("C#", "c#"),
    ("C++", "c++"),
    ("well-known", "well known"),
    ("**Bold**", "bold"),
    ("", ""),
]


@pytest.mark.parametrize(("text", "expected"), NORMALIZE_CASES)
def test_normalize_text(text: str, expected: str) -> None:
    assert normalize_text(text) == expected


# --------------------------------------------------------------------------- exact / normalized


@pytest.mark.parametrize(
    ("cand", "target", "expected"),
    [
        ("Paris", "Paris", True),
        (" Paris\n", "Paris", True),
        ("paris", "Paris", False),
        ("Paris.", "Paris", False),
        ("", "Paris", False),
        ("", "", None),  # two missing answers are undecidable
        ("Paris", "", False),
    ],
)
def test_exact(cand: str, target: str, expected: bool) -> None:
    assert Exact().compare(TASK, cand, target).equivalent is expected


NORMALIZED_CASES = [
    ("paris", "Paris", True),
    ("The Eiffel Tower", "eiffel tower.", True),
    ("U.S.", "US", True),
    ("3.5", "35", False),
    ("-3", "3", False),
    ("3/4", "34", False),
    ("1,000", "1000", True),
    ("A", "a", True),
    ("A", "B", False),
    ("an apple", "apple", True),
    ("", "x", False),
    ("   ", "x", False),
    ("...", "?", False),
    ("...", "...", True),
    ("Paris, France", "Paris France", True),
    ("**Paris**", "Paris", True),
]


@pytest.mark.parametrize(("cand", "target", "expected"), NORMALIZED_CASES)
def test_normalized(cand: str, target: str, expected: bool) -> None:
    j = Normalized().compare(TASK, cand, target)
    assert j.equivalent is expected
    assert j.comparator == "normalized"


# --------------------------------------------------------------------------- parse_number

PARSE_CASES = [
    ("42", Fraction(42), False, None, ""),
    ("-3", Fraction(-3), False, None, ""),
    ("−3", Fraction(-3), False, None, ""),
    ("+3", Fraction(3), False, None, ""),
    ("1,234.5", Fraction(24690, 20), False, None, ""),
    ("1,234,567", Fraction(1234567), False, None, ""),
    ("0.5", Fraction(1, 2), False, None, ""),
    (".5", Fraction(1, 2), False, None, ""),
    ("42.", Fraction(42), False, None, ""),
    ("3/4", Fraction(3, 4), False, None, ""),
    ("3 / 4", Fraction(3, 4), False, None, ""),
    ("\\frac{1}{2}", Fraction(1, 2), False, None, ""),
    ("$\\dfrac{3}{4}$", Fraction(3, 4), False, None, ""),
    ("1e-3", Fraction(1, 1000), False, None, ""),
    ("2.5E3", Fraction(2500), False, None, ""),
    ("1.5 x 10^3", Fraction(1500), False, None, ""),
    ("1.5 \\times 10^{-2}", Fraction(15, 1000), False, None, ""),
    ("50%", Fraction(50), True, None, ""),
    ("45 percent", Fraction(45), True, None, ""),
    ("5\\%", Fraction(5), True, None, ""),
    ("$12", Fraction(12), False, "USD", ""),
    ("$1,200.50", Fraction(120050, 100), False, "USD", ""),
    ("-$5", Fraction(-5), False, "USD", ""),
    ("$-5", Fraction(-5), False, "USD", ""),
    ("€12", Fraction(12), False, "EUR", ""),
    ("12 USD", Fraction(12), False, "USD", ""),
    ("12 dollars", Fraction(12), False, "USD", ""),
    ("12 apples", Fraction(12), False, None, "apples"),
    ("9.8 m/s^2", Fraction(98, 10), False, None, "m/s^2"),
    ("12 km/h", Fraction(12), False, None, "km/h"),
    ("x = 7", Fraction(7), False, None, ""),
    ("approximately 7", Fraction(7), False, None, ""),
    ("**42**", Fraction(42), False, None, ""),
    ("\\boxed{42}", Fraction(42), False, None, ""),
]


@pytest.mark.parametrize(("text", "value", "pct", "cur", "unit"), PARSE_CASES)
def test_parse_number(text: str, value: Fraction, pct: bool, cur: str | None, unit: str) -> None:
    p = parse_number(text)
    assert p is not None, text
    assert (p.value, p.percent, p.currency, p.unit) == (value, pct, cur, unit)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Paris",
        "12 or 13",
        "between 3 and 4",
        "1,5",  # European decimal comma: refuse to guess
        "1 1/2",  # mixed number: ambiguous
        "3/0",
        "1e999",
        "+-3",
        "$12 €",
        "12 apples and 3 pears",
        "twelve",
        "NaN",
        "inf",
        "five words of unit text here",
    ],
)
def test_parse_number_rejects(text: str) -> None:
    assert parse_number(text) is None


# --------------------------------------------------------------------------- numeric

NUMERIC_CASES = [
    ("42", "42", True),
    ("42.0", "42", True),
    ("42", "43", False),
    ("1,234", "1234", True),
    ("1/2", "0.5", True),
    ("1/3", "0.3333333", True),  # within rel_tol 1e-6
    ("1/3", "0.33", False),
    ("1e-3", "0.001", True),
    ("$12", "12", True),
    ("12 apples", "12", True),
    ("12 apples", "12 apple", True),
    ("12 km", "12 m", False),
    ("$12", "€12", False),
    ("12 USD", "$12", True),
    ("-5", "5", False),
    ("−5", "-5", True),
    ("50%", "0.5", True),  # percent="either" default
    ("50%", "50", True),
    ("50%", "50 %", True),
    ("50%", "5", False),
    ("0.5", "50%", True),
    ("123456789012345678901", "123456789012345678900", False),  # integers: exact
    ("123456789012345678901.0", "123456789012345678900", True),  # rel_tol 1e-6
    ("1e-12", "0", True),  # abs_tol 1e-9
    ("1e-6", "0", False),
    ("\\boxed{17}", "17", True),
    ("Paris", "paris", True),  # fallback to normalized text
    ("Paris", "London", False),
    ("12 or 13", "12", False),
]


@pytest.mark.parametrize(("cand", "target", "expected"), NUMERIC_CASES)
def test_numeric(cand: str, target: str, expected: bool) -> None:
    assert Numeric().compare(TASK, cand, target).equivalent is expected


@pytest.mark.parametrize(
    ("mode", "cand", "target", "expected"),
    [
        ("ratio", "50%", "0.5", True),
        ("ratio", "50%", "50", False),
        ("strict", "50%", "0.5", False),
        ("strict", "50%", "50", False),
        ("strict", "50%", "50%", True),
        ("either", "25%", "0.25", True),
    ],
)
def test_numeric_percent_modes(mode: str, cand: str, target: str, expected: bool) -> None:
    assert Numeric(percent=mode).compare(TASK, cand, target).equivalent is expected


def test_numeric_tolerances() -> None:
    assert Numeric(rel_tol=0.01).compare(TASK, "101.0", "100").equivalent is True
    assert Numeric(rel_tol=0.001).compare(TASK, "101.0", "100").equivalent is False
    # two integers compare exactly whatever the tolerances
    assert Numeric(rel_tol=0.01).compare(TASK, "101", "100").equivalent is False
    assert Numeric(rel_tol=0, abs_tol=0.5).compare(TASK, "3.4", "3").equivalent is True
    assert Numeric(rel_tol=0, abs_tol=0).compare(TASK, "0.1", "1/10").equivalent is True


def test_numeric_no_fallback() -> None:
    j = Numeric(fallback_text=False).compare(TASK, "Paris", "12")
    assert j.equivalent is None
    assert "candidate" in j.detail["reason"]
    j = Numeric(fallback_text=False).compare(TASK, "12", "twelve")
    assert j.equivalent is None
    assert "target" in j.detail["reason"]


def test_numeric_empty_candidate_is_false() -> None:
    j = Numeric(fallback_text=False).compare(TASK, "", "12")
    assert j.equivalent is False


def test_numeric_detail_and_bad_config() -> None:
    j = Numeric().compare(TASK, "12 km", "12 m")
    assert j.detail["reason"] == "unit mismatch"
    with pytest.raises(ConfigError):
        Numeric(percent="sometimes")
    with pytest.raises(ConfigError):
        Numeric(rel_tol=-1)


# ------------------------------------------------------------------ choice / contains / regex


@pytest.mark.parametrize(
    ("cand", "target", "expected"),
    [
        ("Answer: (C)", "C", True),
        ("the answer is c.", "C", True),
        ("C", "(C)", True),
        ("B", "C", False),
        ("A man said B", "B", True),
        ("Paris", "C", False),
        ("", "C", False),
        ("C", "Paris", None),
    ],
)
def test_choice(cand: str, target: str, expected: bool | None) -> None:
    assert ChoiceComparator().compare(TASK, cand, target).equivalent is expected


@pytest.mark.parametrize(
    ("cand", "target", "expected"),
    [
        ("The capital is Paris.", "paris", True),
        ("It is 42 exactly", "42", True),
        ("It is 420", "42", False),
        ("New York City", "york city", True),
        ("New York", "York City", False),
        ("anything", "", None),
        ("", "", None),
        ("...", "...", True),
        ("...", "?", False),
        ("", "x", False),
    ],
)
def test_contains(cand: str, target: str, expected: bool | None) -> None:
    assert Contains().compare(TASK, cand, target).equivalent is expected


@pytest.mark.parametrize(
    ("kwargs", "cand", "target", "expected"),
    [
        ({}, "42", r"4\d", True),
        ({}, " 42 ", r"4\d", True),
        ({}, "142", r"4\d", False),
        ({"mode": "search"}, "142", r"4\d", True),
        ({}, "PARIS", "paris", False),
        ({"ignore_case": True}, "PARIS", "paris", True),
        ({}, "", ".*", False),
        ({}, "x", "(", None),
    ],
)
def test_regex(kwargs: dict, cand: str, target: str, expected: bool | None) -> None:
    assert RegexComparator(**kwargs).compare(TASK, cand, target).equivalent is expected


# --------------------------------------------------------------------------- judge

VERDICT_CASES = [
    ("They match.\nVERDICT: EQUIVALENT", True),
    ("VERDICT: DIFFERENT", False),
    ("verdict: equivalent", True),
    ("**Verdict:** DIFFERENT", False),
    ("VERDICT: NOT EQUIVALENT", False),
    ("Verdict - Equivalent", True),
    ("VERDICT: `EQUIVALENT`", True),
    ("VERDICT: DIFFERENT ... actually VERDICT: EQUIVALENT", True),
    ("VERDICT: EQUIVALENT\nOn reflection, VERDICT: DIFFERENT", False),
    ("The answers are equivalent.", None),
    ("VERDICT: maybe", None),
    ("", None),
]


@pytest.mark.parametrize(("text", "expected"), VERDICT_CASES)
def test_parse_verdict(text: str, expected: bool | None) -> None:
    assert parse_verdict(text) is expected


def test_judge_equivalent_records_call_and_tags() -> None:
    be = FakeBackend("Same city.\nVERDICT: EQUIVALENT")
    j = JudgeComparator(be).compare(TASK, "Paris, the French capital", "Paris")
    assert j.equivalent is True
    assert j.comparator == "judge"
    assert len(j.calls) == 1 and j.calls[0].text.endswith("EQUIVALENT")
    assert j.detail["verdict"] == "EQUIVALENT"
    req = be.requests[0]
    assert req.tags["role"] == "judge"
    assert req.tags["task_id"] == "t1"
    assert TASK.prompt in req.prompt
    assert "Paris, the French capital" in req.prompt


def test_judge_different() -> None:
    j = JudgeComparator(FakeBackend("VERDICT: DIFFERENT")).compare(TASK, "Lyon", "Paris")
    assert j.equivalent is False
    assert len(j.calls) == 1


def test_judge_unparseable_is_none_with_call() -> None:
    j = JudgeComparator(FakeBackend("I am not sure.")).compare(TASK, "Lyon", "Paris")
    assert j.equivalent is None
    assert "unparseable" in j.detail["error"]
    assert len(j.calls) == 1


def test_judge_backend_error_is_none() -> None:
    be = FakeBackend(error=BackendError("boom", backend="fake:judge"))
    j = JudgeComparator(be).compare(TASK, "Lyon", "Paris")
    assert j.equivalent is None
    assert "boom" in j.detail["error"]
    assert j.calls == ()


def test_judge_shortcut_and_empty() -> None:
    be = FakeBackend("VERDICT: DIFFERENT")
    judge = JudgeComparator(be)
    assert judge.compare(TASK, "paris.", "Paris").equivalent is True
    assert judge.compare(TASK, "", "Paris").equivalent is False
    assert judge.compare(TASK, "", "").equivalent is None
    assert be.requests == []
    no_shortcut = JudgeComparator(be, shortcut=False)
    assert no_shortcut.compare(TASK, "Paris", "Paris").equivalent is False
    assert len(be.requests) == 1


def test_judge_custom_prompt_keeps_other_braces() -> None:
    be = FakeBackend("VERDICT: EQUIVALENT")
    tmpl = 'Q={question} C={candidate} T={target} json={"k": 1} {other}'
    JudgeComparator(be, tmpl, max_tokens=64, system="sys").compare(TASK, "a {b}", "{target}")
    req = be.requests[0]
    assert req.prompt == f'Q={TASK.prompt} C=a {{b}} T={{target}} json={{"k": 1}} {{other}}'
    assert req.max_tokens == 64 and req.system == "sys"


def test_judge_prompt_validation() -> None:
    with pytest.raises(ConfigError):
        JudgeComparator(FakeBackend(), "no placeholders")
    assert "{candidate}" in DEFAULT_JUDGE_PROMPT and "{question}" in DEFAULT_JUDGE_PROMPT


# --------------------------------------------------------------------------- factory


@pytest.mark.parametrize(
    ("spec", "cls"),
    [
        (None, Normalized),
        ({"type": "exact"}, Exact),
        ({"type": "normalized"}, Normalized),
        ({"type": "numeric", "rel_tol": 0.01, "percent": "ratio"}, Numeric),
        ({"type": "choice", "letters": "ABCD"}, ChoiceComparator),
        ({"type": "contains"}, Contains),
        ({"type": "regex", "mode": "search", "ignore_case": True}, RegexComparator),
        ({"type": "judge", "backend": "j", "max_tokens": 100}, JudgeComparator),
        ("numeric", Numeric),
    ],
)
def test_from_spec(spec: object, cls: type) -> None:
    comp = from_spec(spec, backends={"j": FakeBackend()})  # type: ignore[arg-type]
    assert isinstance(comp, cls)
    assert isinstance(comp, Comparator)


def test_from_spec_numeric_options_applied() -> None:
    comp = from_spec({"type": "numeric", "rel_tol": 0.1, "percent": "strict"})
    assert comp.compare(TASK, "105.0", "100").equivalent is True
    assert comp.compare(TASK, "50%", "0.5").equivalent is False


@pytest.mark.parametrize(
    ("spec", "needle"),
    [
        ({"type": "fuzzy"}, "fuzzy"),
        ({}, "None"),
        ({"type": "exact", "strict": True}, "strict"),
        ({"type": "numeric", "tolerance": 0.1}, "tolerance"),
        ({"type": "numeric", "percent": "x"}, "percent"),
        ({"type": "judge"}, "backend"),
        ({"type": "judge", "backend": "missing"}, "missing"),
        ({"type": "judge", "backend": "j", "temp": 0}, "temp"),
        ({"type": "judge", "backend": "j", "prompt": "{candidate} only"}, "target"),
        ({"type": "regex", "mode": "match"}, "mode"),
        (42, "table"),
    ],
)
def test_from_spec_errors(spec: object, needle: str) -> None:
    with pytest.raises(ConfigError, match=needle):
        from_spec(spec, backends={"j": FakeBackend()})  # type: ignore[arg-type]


# --------------------------------------------------------------------------- regressions


@pytest.mark.parametrize(
    "comparator",
    [
        Exact(),
        Normalized(),
        Numeric(),
        Numeric(fallback_text=False),
        ChoiceComparator(),
        Contains(),
        RegexComparator(),
        RegexComparator(mode="search"),
    ],
)
@pytest.mark.parametrize(("cand", "target"), [("", ""), ("  ", "\n"), ("", " ")])
def test_two_empty_answers_are_undecidable(comparator, cand: str, target: str) -> None:
    j = comparator.compare(TASK, cand, target)
    assert j.equivalent is None
    assert j.detail["reason"] == "both empty"


def test_judge_two_empty_answers_undecidable_without_call() -> None:
    be = FakeBackend("VERDICT: EQUIVALENT")
    for judge in (JudgeComparator(be), JudgeComparator(be, shortcut=False)):
        assert judge.compare(TASK, "", " ").equivalent is None
    assert be.requests == []


@pytest.mark.parametrize(
    ("cand", "target", "expected"),
    [
        # two integers: exact, whatever the tolerance
        ("1234567", "1234568", False),
        ("1000000", "1000001", False),
        ("1,000,000", "1000001", False),
        ("2500000", "2500002", False),
        ("1000000", "1,000,000", True),
        ("$1000000", "1000000 dollars", True),
        ("1000000.", "1000001", False),  # trailing sentence period is still an integer
        # either side decimal / scientific / fraction: tolerances apply
        ("1e6", "1000000", True),
        ("1.0e6", "1000001", True),  # rel diff 1e-6
        ("1000000.0", "1000001", True),
        ("1000000", "1000001.0", True),
        ("0.30000000000000004", "0.3", True),
        ("0.1", "1/10", True),
        ("0.333333", "1/3", True),
        ("0.3", "0.31", False),
        # percentages keep their semantics
        ("50%", "0.5", True),
        ("50%", "50", True),
        ("50%", "51", False),
        ("100%", "1", True),
        ("33.3333333%", "1/3", True),
    ],
)
def test_numeric_integers_compare_exactly(cand: str, target: str, expected: bool) -> None:
    assert Numeric().compare(TASK, cand, target).equivalent is expected


def test_parse_number_integer_flag() -> None:
    def flag(s: str) -> bool:
        p = parse_number(s)
        assert p is not None, s
        return p.integer

    assert flag("42") and flag("1,234,567") and flag("$12") and flag("50%") and flag("-3")
    assert not flag("42.0") and not flag("1e6") and not flag("3/4") and not flag(".5")
    assert not flag("1.5 x 10^3")
