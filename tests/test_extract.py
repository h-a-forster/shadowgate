from __future__ import annotations

import pytest

from shadowgate.errors import ConfigError
from shadowgate.extract import (
    Choice,
    FinalLine,
    Identity,
    JsonField,
    LastNumber,
    RegexExtractor,
    find_boxed,
    from_spec,
    is_confidence_line,
    remove_confidence,
    strip_markup,
)

# --------------------------------------------------------------------------- final_line

FINAL_LINE_CASES = [
    ("Reasoning here.\nANSWER: 42", "42"),
    ("Reasoning\nANSWER: 42\nCONFIDENCE: 0.9", "42"),
    ("answer: paris", "paris"),
    ("**Answer:** **42**.", "42"),
    ("**Answer**: Tokyo", "Tokyo"),
    ("- **ANSWER:** C", "C"),
    ("### Answer: Tokyo", "Tokyo"),
    ("> ANSWER: quoted", "quoted"),
    ("answer: `x = 3`", "x = 3"),
    ("ANSWER: ```7```", "7"),
    ("Answer: *Paris*", "Paris"),
    ("Answer: _Paris_", "Paris"),
    ("ANSWER: $\\boxed{42}$", "42"),
    ("ANSWER: \\(x^2\\)", "x^2"),
    ("ANSWER: $$7$$", "7"),
    ("ANSWER: \\text{blue}", "blue"),
    ("ANSWER: $12", "$12"),
    ("ANSWER: $12 and $15", "$12 and $15"),
    ("Final answer: 7", "7"),
    ("Final Answer: $42$", "42"),
    ("ANSWER: 1\nmore work\nANSWER: 2", "2"),
    ("My answer: 5 then ANSWER: 6", "6"),
    # a marker at the start of a line beats one buried in later prose
    ("ANSWER: 42\nNote that my previous answer: 41 was wrong", "42"),
    # rest of the line only
    ("ANSWER: 17\nBecause 10 + 7 = 17", "17"),
    # empty marker line -> next non-empty, non-confidence line
    ("ANSWER:\n\n17\nCONFIDENCE: 0.3", "17"),
    ("ANSWER:\nCONFIDENCE: 0.3\n17", "17"),
    # inline confidence is cut
    ("Answer: 42 CONFIDENCE: 0.9", "42"),
    ("ANSWER: Paris (confidence: 0.8)", "Paris"),
    ("ANSWER: Paris | confidence 0.8", "Paris"),
    ("ANSWER: 7 with 90% confidence", "7"),
    ("ANSWER: 42\n**Confidence:** 0.9", "42"),
    ("ANSWER: 3.", "3"),
    ("ANSWER: 3.14.", "3.14"),
    ("ANSWER: ...", "..."),
    ("ANSWER: snake_case_name", "snake_case_name"),
    ("ANSWER: __init__", "__init__"),
    ("ANSWER: -5", "-5"),
    ("ANSWER:42", "42"),
    ("ANSWER : 42", "42"),
    ("The final answer is $\\boxed{\\frac{1}{2}}$.", "1/2"),
    # fallbacks
    ("so we get \\boxed{17} as the result\nCONFIDENCE: 0.9", "17"),
    ("first \\boxed{1} then \\boxed{2}", "2"),
    ("nested \\boxed{\\frac{a}{b}+{c}}", "a/b+{c}"),
    ("no marker\nlast line here\nConfidence: 80%", "last line here"),
    ("Some text\n```\n", "Some text"),
    ("line one\n- 42\n", "42"),
    ("CONFIDENCE: 0.9", ""),
    ("", ""),
    ("   \n  \n", ""),
    # the word "answer" without a colon is not a marker
    ("The answer is clear\nParis", "Paris"),
    # CRLF line endings
    ("Reasoning\r\nANSWER: 9\r\nCONFIDENCE: 0.5\r\n", "9"),
    # a marker with an empty value and nothing after falls back to boxed/last line
    ("\\boxed{5}\nANSWER:", "5"),
]


@pytest.mark.parametrize(("text", "expected"), FINAL_LINE_CASES)
def test_final_line(text: str, expected: str) -> None:
    assert FinalLine().extract(text) == expected


def test_final_line_custom_prefix() -> None:
    ex = FinalLine(prefix="RESULT:", alt_prefixes=())
    assert ex.extract("ANSWER: 1\nRESULT: 2") == "2"
    assert ex.extract("Final answer: 3") == "Final answer: 3"  # no alt prefix -> last line


def test_final_line_alt_prefix_without_colon() -> None:
    ex = FinalLine(alt_prefixes=("The answer is",))
    assert ex.extract("The answer is 12") == "12"


def test_final_line_bad_prefix() -> None:
    with pytest.raises(ConfigError):
        FinalLine(prefix=":")


@pytest.mark.parametrize("bad", [None, 42, b"ANSWER: 1"])
def test_extractors_never_raise(bad: object) -> None:
    for ex in (FinalLine(), LastNumber(), Choice(), Identity(), JsonField("a")):
        assert isinstance(ex.extract(bad), str)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- last_number

LAST_NUMBER_CASES = [
    ("The total is 1,234.5 dollars", "1234.5"),
    ("-3", "-3"),
    ("x = 3/4", "3/4"),
    ("1e-3", "1e-3"),
    ("1.5E+10 m", "1.5E+10"),
    ("it costs $12", "12"),
    ("about 45% of them", "45%"),
    ("12 %", "12%"),
    ("\u22125", "-5"),
    ("\uff0d5", "-5"),
    ("Answer 42\nCONFIDENCE: 0.95", "42"),
    ("ANSWER: 7 (confidence 0.8)", "7"),
    ("ANSWER: 7 (confidence: 0.8)", "7"),
    ("ANSWER: 7, and I am 90% confident", "7"),
    ("pages 3-5", "5"),
    ("H2O", ""),
    ("v1.2.3", ""),
    ("\\frac{3}{4}", "3/4"),
    ("\\dfrac{-1}{2}", "-1/2"),
    ("+7", "7"),
    ("$-5", "-5"),
    ("-$5", "-5"),
    ("1,2345", "2345"),
    ("1,2,3", "3"),
    ("is 0.5.", "0.5"),
    (".5", ".5"),
    ("1{,}000", "1000"),
    ("no numbers", ""),
    ("", ""),
    ("first 10 then 20 then 30", "30"),
    ("１２３", "123"),  # full-width digits via NFKC
    ("3/0", "3/0"),  # extraction does not evaluate
]


@pytest.mark.parametrize(("text", "expected"), LAST_NUMBER_CASES)
def test_last_number(text: str, expected: str) -> None:
    assert LastNumber().extract(text) == expected


# --------------------------------------------------------------------------- choice

CHOICE_CASES = [
    ("(C)", "C"),
    ("C)", "C"),
    ("C", "C"),
    ("c.", "C"),
    ("Answer: c", "C"),
    ("the answer is C.", "C"),
    ("ANSWER: (b)", "B"),
    ("**Answer: E**", "E"),
    ("answer is option B", "B"),
    ("The correct answer would be D.", "D"),
    ("Answer: C is correct", "C"),
    ("\\boxed{D}", "D"),
    ("so it is $\\boxed{\\text{B}}$", "B"),
    ("A", "A"),
    ("I", "I"),
    ("A is correct.", "A"),
    ("The answer is A.", "A"),
    ("C) Paris", "C"),
    ("I think it's D", "D"),
    # article "A" in prose is not a choice
    ("A man walks into a bar. The answer is B", "B"),
    ("A good choice is (D)", "D"),
    ("A careful reading shows that B fits", "B"),
    # lowercase article after "answer is" is not a choice
    ("The answer is a good one, B", "B"),
    ("Answer: I think B", "B"),
    # restated option listing is ignored; prose wins
    ("(A) 1\n(B) 2\n(C) 3\n(D) 4\nSo B is right", "B"),
    ("Options:\nA) x\nB) y\nI pick B", "B"),
    ("Between (A) and (C), (C) is better", "C"),
    ("ANSWER: B\nCONFIDENCE: 0.9", "B"),
    ("U.S.A. is big", ""),
    ("Answer: Paris", ""),
    ("nothing here", ""),
    ("", ""),
    # letters outside A-J are ignored by default
    ("Answer: K", ""),
    ("(Z)", ""),
    ("explicit beats later prose: answer is B. Vitamin C is unrelated", "B"),
]


@pytest.mark.parametrize(("text", "expected"), CHOICE_CASES)
def test_choice(text: str, expected: str) -> None:
    assert Choice().extract(text) == expected


def test_choice_custom_letters() -> None:
    assert Choice("ABCD").extract("Answer: E") == ""
    assert Choice("abcdefghijklmnopqrstuvwxyz").extract("Answer: K") == "K"
    with pytest.raises(ConfigError):
        Choice("1")


# ------------------------------------------------------------------- regex / json / identity


@pytest.mark.parametrize(
    ("kwargs", "text", "expected"),
    [
        ({"pattern": r"#(\d+)"}, "#1 then #2", "2"),
        ({"pattern": r"#(\d+)", "which": "first"}, "#1 then #2", "1"),
        ({"pattern": r"\d+"}, "a 12 b 34", "34"),
        ({"pattern": r"(?P<v>\w+)!", "group": "v"}, "hey! you!", "you"),
        ({"pattern": r"result=(\w+)", "flags": "i"}, "RESULT=ok", "ok"),
        ({"pattern": r"x(\d)?"}, "x", ""),
        ({"pattern": r"nomatch(\d)"}, "abc", ""),
    ],
)
def test_regex(kwargs: dict, text: str, expected: str) -> None:
    assert RegexExtractor(**kwargs).extract(text) == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pattern": "("},
        {"pattern": "a", "flags": "q"},
        {"pattern": "(a)", "group": 2},
        {"pattern": "(a)", "group": "missing"},
        {"pattern": "a", "which": "middle"},
    ],
)
def test_regex_bad_config(kwargs: dict) -> None:
    with pytest.raises(ConfigError):
        RegexExtractor(**kwargs)


@pytest.mark.parametrize(
    ("field", "text", "expected"),
    [
        ("answer", 'Here: {"answer": " 42 "}', "42"),
        ("answer", '```json\n{"answer": 42}\n```', "42"),
        ("answer", '{"answer": true}', "true"),
        ("answer", '{"answer": null}', ""),
        ("answer", '{"other": 1} then {"answer": "B"}', "B"),
        ("result.answer", '{"result": {"answer": "x"}}', "x"),
        ("items.1", '{"items": ["a", "b"]}', "b"),
        ("answer", '{"outer": {"answer": "inner"}}', "inner"),
        ("answer", "not json {broken", ""),
        ("answer", '{"answer": [1, 2]}', "[1, 2]"),
    ],
)
def test_json_field(field: str, text: str, expected: str) -> None:
    assert JsonField(field).extract(text) == expected


def test_identity() -> None:
    assert Identity().extract("  hi there \n") == "hi there"


# --------------------------------------------------------------------------- helpers


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("**42**", "42"),
        ("`42`", "42"),
        ("**`42`**.", "42"),
        ("$\\boxed{42}$.", "42"),
        ("\\boxed{\\text{yes}}", "yes"),
        ("$12", "$12"),
        ("U.S...", "U.S..."),
        ("2*3", "2*3"),
        ("\\(x\\)", "x"),
        ("5\\%", "5%"),
    ],
)
def test_strip_markup(text: str, expected: str) -> None:
    assert strip_markup(text) == expected


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("CONFIDENCE: 0.9", True),
        ("confidence = 0.9", True),
        ("**Confidence:** 80%", True),
        ("- Confidence level: high", True),
        ("My confidence: 0.9", True),
        ("Confidence 0.8", True),
        ("ANSWER: 42", False),
        ("Confidence intervals are useful", False),
    ],
)
def test_is_confidence_line(line: str, expected: bool) -> None:
    assert is_confidence_line(line) is expected


def test_remove_confidence_and_find_boxed() -> None:
    assert remove_confidence("a\nCONFIDENCE: 1\nb (confidence: 0.2)") == "a\nb"
    assert find_boxed("\\boxed{1} x \\fbox{{2}} \\boxed{unclosed") == ["1", "{2}"]


# --------------------------------------------------------------------------- factory


@pytest.mark.parametrize(
    ("spec", "cls"),
    [
        (None, FinalLine),
        ({"type": "final_line"}, FinalLine),
        ({"type": "final_line", "prefix": "RESULT:", "alt_prefixes": ["Out:"]}, FinalLine),
        ({"type": "last_number"}, LastNumber),
        ({"type": "choice", "letters": "ABCD"}, Choice),
        ({"type": "regex", "pattern": "(x)"}, RegexExtractor),
        ({"type": "json_field", "field": "a"}, JsonField),
        ({"type": "identity"}, Identity),
        ("identity", Identity),
    ],
)
def test_from_spec(spec: object, cls: type) -> None:
    assert isinstance(from_spec(spec), cls)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("spec", "needle"),
    [
        ({"type": "nope"}, "nope"),
        ({}, "None"),
        ({"type": "final_line", "prefx": "A:"}, "prefx"),
        ({"type": "identity", "x": 1}, "'x'"),
        ({"type": "regex"}, "pattern"),
        ({"type": "json_field"}, "field"),
        ({"type": "regex", "pattern": "("}, "invalid pattern"),
        (["final_line"], "table"),
    ],
)
def test_from_spec_errors(spec: object, needle: str) -> None:
    with pytest.raises(ConfigError, match=needle):
        from_spec(spec)  # type: ignore[arg-type]


def test_from_spec_options_applied() -> None:
    ex = from_spec({"type": "final_line", "prefix": "RESULT:", "alt_prefixes": ["Out:"]})
    assert ex.extract("Out: 5") == "5"
    assert ex.extract("RESULT: 6") == "6"


# --------------------------------------------------------------------------- regressions


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # answers that merely mention confidence are kept whole
        ("ANSWER: 95% confidence interval", "95% confidence interval"),
        ("ANSWER: [1.2, 3.4] at 95% confidence", "[1.2, 3.4] at 95% confidence"),
        ("ANSWER: vote of no confidence: 3 votes", "vote of no confidence: 3 votes"),
        ("ANSWER: confidence intervals", "confidence intervals"),
        ("ANSWER: Paris (confidence: 0.8) because of X", "Paris (confidence: 0.8) because of X"),
        # a single trailing value is still cut
        ("ANSWER: 42 (confidence: 0.8)", "42"),
        ("ANSWER: 42 | confidence 0.8", "42"),
        ("ANSWER: 42 confidence=85%", "42"),
        ("ANSWER: 42, confidence: 85%.", "42"),
        ("ANSWER: 42 [conf: 0.7]", "42"),
        ("ANSWER: 42 (confidence: high)", "42"),
        ("ANSWER: 42 (confidence 8/10)", "42"),
        ("ANSWER: 42 (90% confidence)", "42"),
        ("ANSWER: 42 with a 90% confidence", "42"),
        ("ANSWER: 42 - confidence - 0.85", "42"),
        # the fallback never returns the bare marker
        ("ANSWER: (confidence: 0.9)", ""),
        ("ANSWER:", ""),
        ("**Answer:**", ""),
        ("The result\nANSWER: (confidence 0.9)", "The result"),
    ],
)
def test_final_line_inline_confidence(text: str, expected: str) -> None:
    assert FinalLine().extract(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("So x = 12, I'm 90% confident", "12"),
        ("ANSWER: 42 (confidence: 0.9)", "42"),
        ("ANSWER: [1.2, 3.4] at 95% confidence", "95%"),
        ("ANSWER: vote of no confidence: 3 votes", "3"),
    ],
)
def test_last_number_inline_confidence(text: str, expected: str) -> None:
    assert LastNumber().extract(text) == expected
