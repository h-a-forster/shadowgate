"""Answer extractors: pull the final answer out of a model's free-form reply.

Every extractor implements :class:`Extractor` (``name`` + ``extract(text) -> str``). Extraction
never raises: when nothing usable is found the result is ``""``.

Built-in types (see :func:`from_spec`):

* ``final_line`` - the text after the last ``ANSWER:`` marker (configurable), cleaned of
  markdown/LaTeX decoration. Falls back to the last ``\\boxed{...}``, then to the last non-empty
  line that is not a ``CONFIDENCE:`` line.
* ``last_number`` - the last number in the text (ignoring confidence lines).
* ``choice`` - a single multiple-choice letter (A-J by default).
* ``regex`` - a capture group of the last (or first) match of a pattern.
* ``json_field`` - a field of the first JSON object in the text that has it.
* ``identity`` - the stripped text.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any, Protocol, runtime_checkable

from shadowgate.errors import ConfigError

__all__ = [
    "Extractor",
    "FinalLine",
    "LastNumber",
    "Choice",
    "RegexExtractor",
    "JsonField",
    "Identity",
    "strip_markup",
    "find_boxed",
    "is_confidence_line",
    "remove_confidence",
    "from_spec",
]

log = logging.getLogger("shadowgate.extract")


@runtime_checkable
class Extractor(Protocol):
    name: str

    def extract(self, text: str) -> str: ...


# --------------------------------------------------------------------------- shared helpers

_UNWRAP_CMDS = ("boxed", "fbox", "text", "textbf", "textit", "mathrm", "mathbf", "mbox", "emph")
_CMD_RE = re.compile(r"\\(" + "|".join(_UNWRAP_CMDS) + r")\s*\{")
_BOXED_RE = re.compile(r"\\(?:boxed|fbox)\s*\{")
_SIMPLE_FRAC_RE = re.compile(r"\\[dt]?frac\s*\{\s*([^{}]+?)\s*\}\s*\{\s*([^{}]+?)\s*\}")
_FRAC_REPL = r"\g<1>/\g<2>"

# A line that only reports confidence: "CONFIDENCE: 0.8", "**Confidence:** 80%",
# "- Confidence level: high", "My confidence: 0.9", "Confidence 0.8",
# "Confidence - 0.8" (hyphen, en dash or em dash).
_CONF_LINE_RE = re.compile(
    r"^[\s>#*_\-`]*(?:(?:my|final|overall|estimated|self[- ]reported)\s+){0,2}"
    r"(?:confidence|conf\.?)\b(?:[^:=\n]{0,25}[:=]|[\s*_]*(?:level|score)?\s*(?:of\s+)?[\d.]"
    # dash separators: "Confidence - 0.85", "Confidence – 85%", "**Confidence** — high"
    r"|[\s*_]*(?:(?:level|score)[\s*_]*)?[-–—][\s*_]*(?:[\d.]|(?:high|medium|low)\b))",
    re.IGNORECASE,
)
# A trailing confidence segment on an answer line. It must end the line and carry exactly one
# value (a number, percentage, ratio or high/medium/low), so answers that merely mention
# confidence ("95% confidence interval", "[1.2, 3.4] at 95% confidence", "vote of no
# confidence: 3 votes") are left alone. Cut: "42 (confidence: 0.9)", "42 | CONFIDENCE 0.9",
# "42 confidence=85%", "42 with 90% confidence", "x = 12, I'm 90% confident".
_CONF_VALUE = r"(?:(?:\d+(?:\.\d*)?|\.\d+)\s*%?(?:\s*/\s*\d+(?:\.\d+)?)?|high|medium|low)"
_CONF_INLINE_RE = re.compile(
    # "<sep> confidence[ level|score][:=-] <value>"
    r"(?:\s*[,;|]\s*|\s+[-–—]\s+|\s*[(\[]\s*|\s+)[*_]*(?:confidence|conf\.?)(?:\s+(?:level|score))?"
    rf"[*_]*\s*(?:[:=\-–—]\s*|of\s+)?[*_]*{_CONF_VALUE}[*_]*\s*[)\]]?"
    r"[\s.]*$"
    # "with [a] <value> confidence|certainty"
    r"|\s*[(\[]?\s*\bwith\s+(?:a\s+)?(?:\d+(?:\.\d+)?\s*%?)\s+(?:confidence|certainty)\b"
    r"\s*[)\]]?[\s.]*$"
    # "<sep> [and] [I'm|I am] <pct> confident|confidence"
    r"|(?:\s*[,;|]\s*|\s*[(\[]\s*|\s+-\s+)(?:and\s+)?(?:i['’]?m\s+|i\s+am\s+)?\d+(?:\.\d+)?\s*%"
    r"\s+(?:confidence|confident)\s*[)\]]?[\s.]*$",
    re.IGNORECASE,
)
_MATH_WRAP_RE = re.compile(r"\$\$(.+)\$\$|\$([^$]+)\$|\\\((.+)\\\)|\\\[(.+)\\\]", re.DOTALL)
_FENCE_ONLY_RE = re.compile(r"^[\s`~\-=*_#>|]*$")
_BULLET_RE = re.compile(r"^\s*(?:#{1,6}\s+|>\s*|[-*+]\s+)")


def _balanced(text: str, open_idx: int) -> tuple[str, int] | None:
    """Return (content, index after the closing brace) for the brace group at ``open_idx``."""
    depth = 0
    for i in range(open_idx, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : i], i + 1
    return None


def _unwrap_commands(s: str) -> str:
    """Replace ``\\boxed{x}``, ``\\text{x}`` ... by ``x`` (balanced braces)."""
    for _ in range(50):
        m = _CMD_RE.search(s)
        if m is None:
            break
        got = _balanced(s, m.end() - 1)
        if got is None:  # unbalanced: drop the command head, keep the rest
            s = s[: m.start()] + s[m.end() :]
            continue
        inner, end = got
        s = s[: m.start()] + inner + s[end:]
    return s


def find_boxed(text: str) -> list[str]:
    """Contents of every ``\\boxed{...}`` / ``\\fbox{...}`` group, in order of appearance."""
    out: list[str] = []
    pos = 0
    while True:
        m = _BOXED_RE.search(text, pos)
        if m is None:
            return out
        got = _balanced(text, m.end() - 1)
        if got is None:
            return out
        out.append(got[0])
        pos = got[1]


def strip_markup(s: str) -> str:
    """Remove decoration around an answer: markdown bold/italics, backticks, LaTeX ``\\boxed``
    and ``\\text`` wrappers, simple ``\\frac{a}{b}`` (-> ``a/b``), ``$...$`` / ``\\(...\\)``
    math delimiters around the whole answer, stray ``**`` at either end, and a single trailing
    period. Applied until nothing changes.

    A lone currency sign (``$12``) is kept: only a ``$`` pair wrapping the whole string is math.
    """
    prev = None
    for _ in range(20):
        if s == prev:
            break
        prev = s
        s = s.strip()
        s = _SIMPLE_FRAC_RE.sub(_FRAC_REPL, _unwrap_commands(s))
        m = _MATH_WRAP_RE.fullmatch(s)
        if m:
            s = next(g for g in m.groups() if g is not None).strip()
        s = re.sub(r"`+([^`]*)`+", r"\1", s)
        s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
        m = re.fullmatch(r"\*([^*]+)\*|_([^_]+)_", s)
        if m:
            s = (m.group(1) if m.group(1) is not None else m.group(2)).strip()
        s = re.sub(r"^\*{2,}|\*{2,}$", "", s).strip()
        s = s.replace("\\%", "%").replace("\\$", "$").replace("\\,", "").replace("\\!", "")
        if s.endswith(".") and not s.endswith(".."):
            s = s[:-1].rstrip()
    return s.strip()


def is_confidence_line(line: str) -> bool:
    """True for a line that reports a confidence score (``CONFIDENCE: 0.8`` and variants)."""
    return bool(_CONF_LINE_RE.match(line))


def _cut_confidence(line: str) -> str:
    return _CONF_INLINE_RE.sub("", line)


def remove_confidence(text: str) -> str:
    """Drop confidence lines and trailing inline confidence segments from ``text``."""
    kept = [_cut_confidence(ln) for ln in text.splitlines() if not is_confidence_line(ln)]
    return "\n".join(kept)


def _safe(fn: Any, text: Any) -> str:
    try:
        if text is None:
            return ""
        return fn(text if isinstance(text, str) else str(text))
    except Exception:  # extraction must never raise
        log.debug("extractor failed", exc_info=True)
        return ""


# --------------------------------------------------------------------------- final_line


def _prefix_pattern(prefix: str) -> str:
    label = prefix.strip().strip("*_ ")
    needs_colon = label.endswith(":")
    label = label.rstrip(":").strip("*_ ")
    if not label:
        raise ConfigError("final_line prefix must contain a label such as 'ANSWER:'")
    words = r"\s+".join(re.escape(w) for w in label.split())
    colon = r"\s*:" if needs_colon else r"\s*:?"
    return rf"(?<![\w])[*_]{{0,3}}{words}[*_]{{0,3}}{colon}[*_]{{0,3}}"


class FinalLine:
    """Text after the last answer marker on its line.

    ``prefix`` (default ``"ANSWER:"``) and each of ``alt_prefixes`` (default
    ``("Final answer:",)``) are matched case-insensitively and tolerate markdown decoration such
    as ``**Answer:**`` or ``**Answer**:``. Markers at the start of a line (after optional
    bullets, ``#`` or ``>``) win over markers inside prose; among those the last one wins.
    Only the rest of that line is used; a trailing confidence segment holding a single value
    (``(confidence: 0.8)``, ``| confidence 0.8``, ``confidence=85%``, ``with 90% confidence``)
    is cut, but text that merely mentions confidence (``95% confidence interval``) is kept.
    When the marker's line is empty the next non-empty, non-confidence line is used.

    Fallback when no marker yields text: the last ``\\boxed{...}``, else the last non-empty
    line that is not a confidence line, a bare code fence / rule, or a bare answer marker.
    """

    name = "final_line"

    def __init__(self, prefix: str = "ANSWER:", alt_prefixes: Iterable[str] = ("Final answer:",)):
        if isinstance(alt_prefixes, str):
            alt_prefixes = (alt_prefixes,)
        self.prefix = prefix
        self.alt_prefixes = tuple(alt_prefixes)
        body = "|".join(_prefix_pattern(p) for p in (prefix, *self.alt_prefixes))
        self._anywhere = re.compile(rf"(?:{body})", re.IGNORECASE)
        self._line_start = re.compile(rf"^[ \t>#\-]*(?:{body})", re.IGNORECASE | re.MULTILINE)
        self._bare_marker = re.compile(rf"[\s>#\-*_]*(?:{body})[\s*_]*", re.IGNORECASE)

    def extract(self, text: str) -> str:
        return _safe(self._extract, text)

    def _from_marker(self, text: str) -> str:
        matches = list(self._line_start.finditer(text)) or list(self._anywhere.finditer(text))
        if not matches:
            return ""
        end = matches[-1].end()
        nl = text.find("\n", end)
        rest = text[end:] if nl < 0 else text[end:nl]
        ans = strip_markup(_cut_confidence(rest))
        if ans or nl < 0:
            return ans
        for line in text[nl + 1 :].splitlines():
            if not line.strip() or is_confidence_line(line):
                continue
            if _FENCE_ONLY_RE.match(line):
                continue
            return strip_markup(_cut_confidence(line))
        return ""

    def _extract(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        ans = self._from_marker(text)
        if ans:
            return ans
        boxed = [b for b in (strip_markup(x) for x in find_boxed(text)) if b]
        if boxed:
            return boxed[-1]
        for line in reversed(text.splitlines()):
            if not line.strip() or is_confidence_line(line) or _FENCE_ONLY_RE.match(line):
                continue
            ans = strip_markup(_BULLET_RE.sub("", _cut_confidence(line)))
            if ans and not self._bare_marker.fullmatch(ans):  # never return the marker itself
                return ans
        return ""


# --------------------------------------------------------------------------- last_number

_MINUS_CHARS = "\u2212\ufe63\uff0d"
_NUM_RE = re.compile(
    r"(?<![\w.])"
    r"(?P<sign>[-+])?"
    r"(?P<cur>[$€£¥₹])?"
    r"(?P<sign2>[-+])?"
    r"(?P<body>(?:\d{1,3}(?:,\d{3})+(?!\d)|\d+)(?:\.\d+)?|\.\d+)"
    r"(?:/(?P<den>\d+)(?![\d.]))?"
    r"(?P<exp>[eE][-+]?\d+)?"
    r"(?P<pct>\s?%)?"
)
_FRAC_LATEX_RE = re.compile(r"\\[dt]?frac\s*\{\s*(-?[\d.,]+)\s*\}\s*\{\s*([\d.,]+)\s*\}")


def _numeric_prep(text: str) -> str:
    s = unicodedata.normalize("NFKC", text)
    for ch in _MINUS_CHARS:
        s = s.replace(ch, "-")
    s = _FRAC_LATEX_RE.sub(r"\1/\2", s)
    return s.replace("{,}", ",").replace("\\%", "%").replace("\\$", "$").replace("\\,", "")


class LastNumber:
    """The last number in the text, ignoring confidence lines.

    Handles signs (including the unicode minus), thousands separators (``1,234.5`` ->
    ``1234.5``), decimals, fractions (``3/4``, ``\\frac{3}{4}``), scientific notation
    (``1e-3``), currency prefixes (``$12`` -> ``12``) and percentages (``45%`` keeps the ``%``
    so a numeric comparator can apply its percent policy). A number glued to a preceding
    letter (``H2O``, ``v2``) or to a dot (``v1.2.3``) is ignored. ``3-5`` yields ``5``
    (the hyphen is a range, not a sign). A leading ``+`` is dropped.
    """

    name = "last_number"

    def extract(self, text: str) -> str:
        return _safe(self._extract, text)

    def _extract(self, text: str) -> str:
        s = _numeric_prep(remove_confidence(text))
        last = None
        for m in _NUM_RE.finditer(s):
            last = m
        if last is None:
            return ""
        if last.group("sign") and last.group("sign2"):
            sign = "-" if (last.group("sign") == "-") != (last.group("sign2") == "-") else ""
        else:
            sign = "-" if "-" in ((last.group("sign") or "") + (last.group("sign2") or "")) else ""
        out = sign + last.group("body").replace(",", "")
        if last.group("den"):
            out += "/" + last.group("den")
        if last.group("exp"):
            out += last.group("exp")
        if last.group("pct"):
            out += "%"
        return out


# --------------------------------------------------------------------------- choice

_CHOICE_EXPLICIT_RE = re.compile(
    r"\banswer\b\s*"
    r"(?:(?:is|would\s+be|should\s+be|must\s+be|will\s+be)\b)?\s*[:=\-]?\s*"
    r"(?:(?:option|choice|letter)\b)?\s*[:\-]?\s*"
    r"(?P<open>[(\[{])?\s*(?P<letter>[A-Za-z])(?![\w'’])\s*(?P<close>[)\]}])?",
    re.IGNORECASE,
)
_LETTER_LINE_RE = re.compile(r"\s*[(\[]?\s*([A-Za-z])\s*[)\]]?\s*[.:]?\s*")
_OPTION_LINE_RE = re.compile(r"\s*[(\[]?([A-Z])[)\].:]\s+\S")
_PAREN_RE = re.compile(r"\(([A-Z])\)")
_STANDALONE_RE = re.compile(r"(?<![\w'’\-.])([A-Z])(?![\w'’\-])(?!\.\w)")
_END_OR_PUNCT_RE = re.compile(r"\s*(?:[.,;:!?)\]]|$)")
_A_VERB_RE = re.compile(r"\s+(?:is|was|seems|looks|appears)\b")


class Choice:
    """A single multiple-choice letter (uppercase), from ``letters`` (default ``A``-``J``).

    Heuristics, in priority order (the first tier that finds a letter wins; within a tier
    the last occurrence wins):

    1. Explicit statements: ``Answer: C``, ``ANSWER: (C)``, ``the answer is c.``,
       ``answer is option B``. A lowercase letter, or ``A``/``I`` (an article/pronoun in prose),
       counts only when bracketed or followed by punctuation or end of line, so
       ``the answer is a man`` is not ``A``.
    2. The last ``\\boxed{...}`` whose content is a single letter.
    3. A line that is only a letter (``C``, ``(C)``, ``C)``, ``c.``), or exactly one line that
       looks like a chosen option (``C) Paris``); several such lines are an option listing
       and are ignored.
    4. The last ``(C)`` in prose, unless the parenthesized letters run in increasing order
       (three or more), which looks like a restated option list.
    5. The last standalone capital letter. ``A`` and ``I`` count only before punctuation /
       end of line (``A`` also before ``is``/``was``/``seems``...), so ``A man ...`` and
       ``I think`` are skipped.

    Confidence lines are ignored. Returns ``""`` when nothing qualifies.
    """

    name = "choice"

    def __init__(self, letters: str = "ABCDEFGHIJ"):
        letters = "".join(dict.fromkeys(letters.upper()))
        if not letters or not all("A" <= c <= "Z" for c in letters):
            raise ConfigError("choice letters must be a non-empty string of letters A-Z")
        self.letters = letters

    def extract(self, text: str) -> str:
        return _safe(self._extract, text)

    def _ok(self, letter: str) -> bool:
        return letter.upper() in self.letters

    def _extract(self, text: str) -> str:
        raw = remove_confidence(text.replace("\r\n", "\n"))
        boxed = [strip_markup(b) for b in find_boxed(raw)]
        s = _unwrap_commands(raw)
        s = s.replace("**", "").replace("`", "").replace("$", "")

        # 1. explicit answer statements
        found = ""
        for m in _CHOICE_EXPLICIT_RE.finditer(s):
            letter = m.group("letter")
            if not self._ok(letter):
                continue
            bracketed = bool(m.group("open")) and bool(m.group("close"))
            plain_upper = letter.isupper() and letter not in "AI"
            if bracketed or plain_upper or _END_OR_PUNCT_RE.match(s, m.end("letter")):
                found = letter.upper()
        if found:
            return found

        # 2. \boxed{C}
        for b in reversed(boxed):
            bm = _LETTER_LINE_RE.fullmatch(b)
            if bm and self._ok(bm.group(1)):
                return bm.group(1).upper()

        # 3. a line that is just a letter, or a single chosen-option line
        lines = [ln for ln in s.splitlines() if ln.strip()]
        for ln in reversed(lines):
            lm = _LETTER_LINE_RE.fullmatch(ln)
            if lm and self._ok(lm.group(1)):
                return lm.group(1).upper()
        options = [om.group(1) for ln in lines if (om := _OPTION_LINE_RE.match(ln))]
        options = [o for o in options if self._ok(o)]
        if len(options) == 1:
            return options[0]

        # 4. "(C)" in prose, unless it looks like a restated option list
        parens = [p for p in _PAREN_RE.findall(s) if self._ok(p)]
        if parens:
            pairs = zip(parens, parens[1:], strict=False)
            listing = len(parens) >= 3 and all(a < b for a, b in pairs)
            if not listing:
                return parens[-1]

        # 5. last standalone capital letter, skipping the article "A" and pronoun "I"
        found = ""
        for m in _STANDALONE_RE.finditer(s):
            letter = m.group(1)
            if not self._ok(letter):
                continue
            if letter in "AI":
                after = m.end(1)
                if not (
                    _END_OR_PUNCT_RE.match(s, after)
                    or (letter == "A" and _A_VERB_RE.match(s, after))
                ):
                    continue
            found = letter
        return found


# ------------------------------------------------------------------- regex / json / identity

_FLAG_MAP = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL, "x": re.VERBOSE}


class RegexExtractor:
    """Capture ``group`` of the last (``which="last"``, default) or first match of ``pattern``.

    ``group`` defaults to 1 when the pattern has a capture group, else 0. ``flags`` is a string
    of ``i``, ``m``, ``s``, ``x``. The result is stripped.
    """

    name = "regex"

    def __init__(
        self, pattern: str, group: int | str | None = None, flags: str = "", which: str = "last"
    ):
        flag_val = 0
        for f in flags:
            if f not in _FLAG_MAP:
                raise ConfigError(f"regex extractor: unknown flag {f!r} (use i, m, s, x)")
            flag_val |= _FLAG_MAP[f]
        try:
            self.pattern = re.compile(pattern, flag_val)
        except re.error as exc:
            raise ConfigError(f"regex extractor: invalid pattern {pattern!r}: {exc}") from exc
        if which not in ("first", "last"):
            raise ConfigError("regex extractor: which must be 'first' or 'last'")
        if group is None:
            group = 1 if self.pattern.groups else 0
        if isinstance(group, int) and group > self.pattern.groups:
            raise ConfigError(f"regex extractor: pattern has no group {group}")
        if isinstance(group, str) and group not in self.pattern.groupindex:
            raise ConfigError(f"regex extractor: pattern has no group named {group!r}")
        self.group = group
        self.which = which

    def extract(self, text: str) -> str:
        return _safe(self._extract, text)

    def _extract(self, text: str) -> str:
        if self.which == "first":
            m = self.pattern.search(text)
        else:
            m = None
            for m in self.pattern.finditer(text):  # noqa: B007 - keep the last match
                pass
        if m is None:
            return ""
        return (m.group(self.group) or "").strip()


class JsonField:
    """Value of ``field`` (dotted path, e.g. ``"result.answer"``) in the first JSON object of
    the text that contains it. Strings are returned stripped; other values as compact JSON
    (``42``, ``true``, ``[1, 2]``); ``null`` or missing -> ``""``.
    """

    name = "json_field"

    def __init__(self, field: str):
        if not isinstance(field, str) or not field:
            raise ConfigError("json_field extractor needs a non-empty 'field'")
        self.field = field
        self._path = field.split(".")

    def extract(self, text: str) -> str:
        return _safe(self._extract, text)

    def _lookup(self, obj: Any) -> tuple[bool, Any]:
        for key in self._path:
            if isinstance(obj, dict) and key in obj:
                obj = obj[key]
            elif isinstance(obj, list) and key.isdigit() and int(key) < len(obj):
                obj = obj[int(key)]
            else:
                return False, None
        return True, obj

    def _extract(self, text: str) -> str:
        decoder = json.JSONDecoder()
        pos = 0
        while True:
            start = text.find("{", pos)
            if start < 0:
                return ""
            try:
                obj, end = decoder.raw_decode(text, start)
            except ValueError:
                pos = start + 1
                continue
            if isinstance(obj, dict):
                ok, val = self._lookup(obj)
                if ok:
                    if val is None:
                        return ""
                    if isinstance(val, str):
                        return val.strip()
                    return json.dumps(val, ensure_ascii=False)
            pos = start + 1  # nested objects may carry the field


class Identity:
    """The whole text, stripped."""

    name = "identity"

    def extract(self, text: str) -> str:
        return _safe(str.strip, text)


# --------------------------------------------------------------------------- factory

_TYPES: dict[str, tuple[type, frozenset[str]]] = {
    "final_line": (FinalLine, frozenset({"prefix", "alt_prefixes"})),
    "last_number": (LastNumber, frozenset()),
    "choice": (Choice, frozenset({"letters"})),
    "regex": (RegexExtractor, frozenset({"pattern", "group", "flags", "which"})),
    "json_field": (JsonField, frozenset({"field"})),
    "identity": (Identity, frozenset()),
}
_REQUIRED = {"regex": ("pattern",), "json_field": ("field",)}


def from_spec(spec: Mapping[str, Any] | str | None) -> Extractor:
    """Build an extractor from ``{"type": ..., **options}``; ``None`` -> ``FinalLine()``.

    A bare string is shorthand for ``{"type": <string>}``. Unknown types or keys raise
    :class:`~shadowgate.errors.ConfigError`.
    """
    if spec is None:
        return FinalLine()
    if isinstance(spec, str):
        spec = {"type": spec}
    if not isinstance(spec, Mapping):
        raise ConfigError(f"extractor spec must be a table, got {type(spec).__name__}")
    kind = spec.get("type")
    if kind not in _TYPES:
        raise ConfigError(
            f"unknown extractor type {kind!r}; expected one of {', '.join(sorted(_TYPES))}"
        )
    cls, allowed = _TYPES[kind]
    opts = {k: v for k, v in spec.items() if k != "type"}
    for key in opts:
        if key not in allowed:
            raise ConfigError(f"unknown key {key!r} for extractor type {kind!r}")
    for key in _REQUIRED.get(kind, ()):
        if key not in opts:
            raise ConfigError(f"extractor type {kind!r} requires key {key!r}")
    if "alt_prefixes" in opts and isinstance(opts["alt_prefixes"], list):
        opts["alt_prefixes"] = tuple(opts["alt_prefixes"])
    try:
        return cls(**opts)
    except TypeError as exc:
        raise ConfigError(f"invalid options for extractor type {kind!r}: {exc}") from exc
