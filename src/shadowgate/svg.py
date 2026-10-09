"""Dependency-free SVG charts.

Pure string building with deterministic output. Charts are standalone SVG documents that scale
to their container (``width="100%"`` plus a ``viewBox``), carry ``role="img"``, a ``<title>`` and
an ``aria-label``, and take their colours from CSS custom properties (with hex fallbacks) so the
embedding page controls light and dark themes. Text and axes use ``currentColor``.

Public API:

* :class:`Series`, :class:`Marker`, :class:`ErrorBar`, :class:`RefLine`, :class:`Diagonal`
* :func:`xy_chart` (lines, steps and scatter on shared axes), :func:`line_chart`,
  :func:`scatter`, :func:`bar_with_ci`
* :func:`nice_ticks`, :func:`log_ticks`, :func:`format_value`
* :data:`CHART_CSS` (recommended variables for light and dark themes), :data:`PALETTE`
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "CHART_CSS",
    "MAX_SCATTER_POINTS",
    "PALETTE",
    "Diagonal",
    "ErrorBar",
    "Marker",
    "RefLine",
    "Series",
    "bar_with_ci",
    "format_value",
    "line_chart",
    "log_ticks",
    "nice_ticks",
    "scatter",
    "xy_chart",
]

SeriesKind = Literal["line", "scatter", "step"]
ValueFormat = Literal["number", "percent", "currency"]
Shape = Literal["circle", "square", "diamond", "triangle", "triangle-down", "star", "cross", "x"]
Style = int | str | None
LegendPos = Literal["auto", "right", "bottom", "none"]

MAX_SCATTER_POINTS = 2000
_MAX_LINE_POINTS = 5000

# Okabe-Ito based categorical palette (light theme fallbacks); dark values live in CHART_CSS.
PALETTE: tuple[str, ...] = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#6B7280",  # grey
    "#8C6D31",  # brown
)
_PALETTE_DARK: tuple[str, ...] = (
    "#56B4E9",
    "#FF8A3D",
    "#2FD3A0",
    "#E9A3CB",
    "#F0C04A",
    "#9AD4F5",
    "#A1A8B3",
    "#C9A86A",
)
_ROLES_LIGHT: dict[str, str] = {
    "muted": "#9CA3AF",
    "accent": "#D55E00",
    "baseline": "#1F2937",
    "ref": "#CC79A7",
    "bg": "#FFFFFF",
}
_ROLES_DARK: dict[str, str] = {
    "muted": "#6B7280",
    "accent": "#FF8A3D",
    "baseline": "#E5E7EB",
    "ref": "#E9A3CB",
    "bg": "#111827",
}
_FALLBACK: dict[str, str] = {f"series-{i + 1}": c for i, c in enumerate(PALETTE)} | _ROLES_LIGHT


def _css_block(colors: Iterable[tuple[str, str]], indent: str) -> str:
    return "\n".join(f"{indent}--sg-{k}: {v};" for k, v in colors)


_LIGHT_VARS = [(f"series-{i + 1}", c) for i, c in enumerate(PALETTE)] + list(_ROLES_LIGHT.items())
_DARK_VARS = [(f"series-{i + 1}", c) for i, c in enumerate(_PALETTE_DARK)] + list(
    _ROLES_DARK.items()
)

CHART_CSS: str = (
    "/* shadowgate chart theme: set --sg-bg to the page background behind charts. */\n"
    ".sg-chart {\n"
    + _css_block(_LIGHT_VARS, "  ")
    + "\n  display: block;\n  max-width: 100%;\n  height: auto;\n  overflow: visible;\n}\n"
    "@media (prefers-color-scheme: dark) {\n"
    '  :root:not([data-theme="light"]) .sg-chart {\n'
    + _css_block(_DARK_VARS, "    ")
    + "\n  }\n}\n"
    ':root[data-theme="dark"] .sg-chart {\n' + _css_block(_DARK_VARS, "  ") + "\n}\n"
)

_FONT = "system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"
_CHAR_W = 6.3  # approximate glyph advance at 11px
_DASH = "6 4"


# --------------------------------------------------------------------------- data classes


@dataclass(frozen=True)
class Series:
    """A named set of (x, y) points.

    ``kind`` None means the chart default (``line`` for :func:`line_chart`/:func:`xy_chart`,
    ``scatter`` for :func:`scatter`). ``style`` is a palette index (1-8), a role name
    (``"muted"``, ``"accent"``, ``"baseline"``, ``"ref"``), ``"series-N"``, or None for the next
    palette colour. Points with None/NaN/inf coordinates are skipped (and break lines).
    """

    name: str
    points: Sequence[tuple[float | None, float | None]]
    kind: SeriesKind | None = None
    style: Style = None
    dashed: bool = False
    width: float = 2.0
    marker: Shape = "circle"
    size: float | None = None
    opacity: float | None = None
    show_points: bool = False
    in_legend: bool = True


@dataclass(frozen=True)
class Marker:
    """A single labelled point drawn with a distinct shape (baselines, recommended point)."""

    x: float | None
    y: float | None
    label: str = ""
    shape: Shape = "diamond"
    style: Style = "baseline"
    emphasis: bool = False
    in_legend: bool = False


@dataclass(frozen=True)
class ErrorBar:
    """A point with a vertical interval ``[lo, hi]`` at ``x``."""

    x: float | None
    y: float | None
    lo: float | None
    hi: float | None
    style: Style = 1
    show_point: bool = True


@dataclass(frozen=True)
class RefLine:
    """A labelled horizontal (``axis="y"``) or vertical (``axis="x"``) reference line."""

    axis: Literal["x", "y"]
    value: float
    label: str = ""
    style: Style = "ref"
    dashed: bool = True


@dataclass(frozen=True)
class Diagonal:
    """The ``y = x`` line (e.g. perfect calibration)."""

    label: str = "y = x"
    dashed: bool = True


# --------------------------------------------------------------------------- formatting


def _f(v: float) -> str:
    """Stable coordinate formatting (2 decimals, no trailing zeros, no negative zero)."""
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _esc(s: object) -> str:
    text = str(s)
    text = "".join(ch for ch in text if ch in "\t\n\r" or ord(ch) >= 0x20)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _trunc(s: str, n: int) -> str:
    n = max(n, 2)
    return s if len(s) <= n else s[: n - 1] + "…"


def _num(v: object) -> float | None:
    if v is None or isinstance(v, str):
        return None
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _decimals_for_step(step: float) -> int:
    if not step > 0 or not math.isfinite(step):
        return 0
    for dec in range(11):
        if abs(round(step, dec) - step) <= 1e-9 * step:
            return dec
    return 10


def _sig_decimals(v: float, sig: int) -> int:
    if v == 0:
        return 0
    return max(0, sig - 1 - math.floor(math.log10(abs(v))))


def _fixed(v: float, dec: int, *, strip: bool = False, group: bool = True) -> str:
    s = f"{v:,.{dec}f}" if group else f"{v:.{dec}f}"
    if strip and "." in s:
        s = s.rstrip("0").rstrip(".")
    if s.startswith("-") and not any(c in "123456789" for c in s):
        s = s[1:]
    return s


def format_value(v: float, fmt: ValueFormat = "number", *, step: float | None = None) -> str:
    """Format a value for display.

    ``fmt``: ``"number"``, ``"percent"`` (fractions: 0.25 -> "25%") or ``"currency"``
    ("$0.0012" style smart precision). With ``step`` (a tick spacing) all values sharing that
    step get the same number of decimals.
    """
    x = _num(v)
    if x is None:
        return "n/a"
    if fmt == "percent":
        p = x * 100.0
        if step is not None:
            return _fixed(p, _decimals_for_step(step * 100.0)) + "%"
        if p == 0 or abs(p) >= 10:
            dec = 0
        elif abs(p) >= 1:
            dec = 1
        else:
            dec = min(_sig_decimals(p, 2), 6)
        return _fixed(p, dec, strip=True) + "%"
    if fmt == "currency":
        sign = "-" if x < 0 else ""
        a = abs(x)
        if step is not None:
            dec = _decimals_for_step(step)
            if dec == 1:
                dec = 2
            body = _fixed(a, dec)
        elif a == 0:
            body = "0"
        elif a >= 1:
            body = _fixed(a, 2)
        else:
            body = _fixed(a, max(2, min(_sig_decimals(a, 2), 10)))
        if not any(c in "123456789" for c in body):
            sign = ""
        return f"{sign}${body}"
    if step is not None:
        return _fixed(x, _decimals_for_step(step))
    if x == 0:
        return "0"
    return _fixed(x, min(_sig_decimals(x, 3), 8), strip=True)


# --------------------------------------------------------------------------- ticks


def _clean(v: float, dec: int) -> float:
    r = round(v, dec)
    return 0.0 if r == 0 else r


def nice_ticks(lo: float, hi: float, target: int = 5) -> list[float]:
    """Evenly spaced 1-2-5 ticks covering ``[lo, hi]`` (first <= lo, last >= hi)."""
    a, b = _num(lo), _num(hi)
    if a is None or b is None:
        raise ValueError("nice_ticks needs finite bounds")
    if b < a:
        a, b = b, a
    if b - a <= 1e-12 * max(1.0, abs(a), abs(b)):
        pad = abs(a) * 0.1 if a != 0 else 1.0
        a, b = a - pad, b + pad
    raw = (b - a) / max(1, target)
    exp = math.floor(math.log10(raw))
    mag = 10.0**exp
    step = 10.0 * mag
    for m in (1.0, 2.0, 5.0, 10.0):
        if m * mag >= raw * (1 - 1e-9):
            step = m * mag
            break
    dec = max(0, -exp) + 1
    i0 = math.floor(a / step + 1e-9)
    i1 = math.ceil(b / step - 1e-9)
    return [_clean(i * step, dec) for i in range(i0, i1 + 1)]


def log_ticks(lo: float, hi: float, max_ticks: int = 8) -> list[float]:
    """Ticks for a log axis within ``[lo, hi]`` (both > 0): powers of ten, else 1-2-5 steps."""
    a, b = _num(lo), _num(hi)
    if a is None or b is None or a <= 0 or b <= 0:
        raise ValueError("log_ticks needs finite positive bounds")
    if b < a:
        a, b = b, a
    k0 = math.floor(math.log10(a)) - 1
    k1 = math.ceil(math.log10(b)) + 1
    eps = 1e-9

    def candidates(mults: Sequence[int]) -> list[float]:
        out = []
        for k in range(k0, k1 + 1):
            for m in mults:
                v = float(f"{m}e{k}")
                if a * (1 - eps) <= v <= b * (1 + eps):
                    out.append(v)
        return out

    decades = candidates((1,))
    if len(decades) >= 3:
        stride = max(1, math.ceil(len(decades) / max_ticks))
        return decades[::stride]
    for mults in ((1, 2, 5), (1, 2, 3, 4, 5, 6, 7, 8, 9)):
        ticks = candidates(mults)
        if 3 <= len(ticks) <= max_ticks:
            return ticks
        if len(ticks) > max_ticks:
            stride = math.ceil(len(ticks) / max_ticks)
            return ticks[::stride]
    lin = [t for t in nice_ticks(a, b, 4) if t > 0 and a * (1 - eps) <= t <= b * (1 + eps)]
    return lin if len(lin) >= 2 else [a, b]


# --------------------------------------------------------------------------- scales & paint


@dataclass(frozen=True)
class _Scale:
    lo: float
    hi: float
    p0: float
    p1: float
    log: bool = False

    def __call__(self, v: float) -> float:
        if self.log:
            t = (math.log10(v) - math.log10(self.lo)) / (math.log10(self.hi) - math.log10(self.lo))
        else:
            t = (v - self.lo) / (self.hi - self.lo)
        return self.p0 + t * (self.p1 - self.p0)

    def contains(self, v: float) -> bool:
        tol = 1e-9 * max(1.0, abs(self.lo), abs(self.hi))
        return self.lo - tol <= v <= self.hi + tol


def _color_key(style: Style, auto: int) -> str:
    if style is None:
        return f"series-{auto % len(PALETTE) + 1}"
    if isinstance(style, bool):
        raise ValueError(f"invalid style: {style!r}")
    if isinstance(style, int):
        return f"series-{(style - 1) % len(PALETTE) + 1}"
    if style in _FALLBACK:
        return style
    roles = ", ".join(sorted(_ROLES_LIGHT))
    raise ValueError(f"unknown style {style!r}; use 1-8, 'series-N' or one of: {roles}")


def _paint(fill: str | None = None, stroke: str | None = None) -> str:
    """Attributes for colour keys: hex fallback attribute plus a CSS ``var()`` override."""
    attrs: list[str] = []
    styles: list[str] = []
    for prop, key in (("fill", fill), ("stroke", stroke)):
        if key is None:
            continue
        if key in _FALLBACK:
            hx = _FALLBACK[key]
            attrs.append(f'{prop}="{hx}"')
            styles.append(f"{prop}:var(--sg-{key},{hx})")
        else:
            attrs.append(f'{prop}="{key}"')
    if styles:
        attrs.append(f'style="{";".join(styles)}"')
    return " ".join(attrs)


def _text(
    x: float,
    y: float,
    s: str,
    *,
    anchor: str = "start",
    size: float = 11,
    bold: bool = False,
    halo: bool = False,
    extra: str = "",
) -> str:
    parts = [f'<text x="{_f(x)}" y="{_f(y)}" fill="currentColor"']
    if anchor != "start":
        parts.append(f' text-anchor="{anchor}"')
    if size != 11:
        parts.append(f' font-size="{_f(size)}"')
    if bold:
        parts.append(' font-weight="600"')
    if halo:
        parts.append(
            ' stroke="#FFFFFF" stroke-width="3" stroke-linejoin="round" paint-order="stroke"'
            ' style="stroke:var(--sg-bg,#FFFFFF)"'
        )
    if extra:
        parts.append(" " + extra)
    parts.append(f">{_esc(s)}</text>")
    return "".join(parts)


def _shape(shape: str, cx: float, cy: float, r: float, paint: str, extra: str = "") -> str:
    ex = f" {extra}" if extra else ""
    if shape == "circle":
        return f'<circle cx="{_f(cx)}" cy="{_f(cy)}" r="{_f(r)}" {paint}{ex}/>'
    if shape == "square":
        s = r * 0.9
        d = f"M{_f(cx - s)} {_f(cy - s)}H{_f(cx + s)}V{_f(cy + s)}H{_f(cx - s)}Z"
    elif shape == "diamond":
        s = r * 1.25
        d = (
            f"M{_f(cx)} {_f(cy - s)}L{_f(cx + s)} {_f(cy)}"
            f"L{_f(cx)} {_f(cy + s)}L{_f(cx - s)} {_f(cy)}Z"
        )
    elif shape in ("triangle", "triangle-down"):
        s = r * 1.25
        sgn = 1 if shape == "triangle" else -1
        d = (
            f"M{_f(cx)} {_f(cy - sgn * s)}L{_f(cx + s)} {_f(cy + sgn * s * 0.8)}"
            f"L{_f(cx - s)} {_f(cy + sgn * s * 0.8)}Z"
        )
    elif shape == "star":
        pts = []
        for i in range(10):
            rr = r * 1.45 if i % 2 == 0 else r * 0.62
            ang = -math.pi / 2 + i * math.pi / 5
            pts.append(f"{_f(cx + rr * math.cos(ang))} {_f(cy + rr * math.sin(ang))}")
        d = "M" + "L".join(pts) + "Z"
    elif shape in ("cross", "x"):
        s = r * 1.1
        w = r * 0.38
        if shape == "cross":
            d = (
                f"M{_f(cx - w)} {_f(cy - s)}H{_f(cx + w)}V{_f(cy - w)}H{_f(cx + s)}V{_f(cy + w)}"
                f"H{_f(cx + w)}V{_f(cy + s)}H{_f(cx - w)}V{_f(cy + w)}H{_f(cx - s)}V{_f(cy - w)}"
                f"H{_f(cx - w)}Z"
            )
        else:
            k = s * 0.8
            d = (
                f"M{_f(cx - k)} {_f(cy - k)}L{_f(cx + k)} {_f(cy + k)}"
                f"M{_f(cx + k)} {_f(cy - k)}L{_f(cx - k)} {_f(cy + k)}"
            )
            return (
                f'<path d="{d}" fill="none" stroke-width="{_f(w * 2)}" stroke-linecap="round" '
                f"{paint.replace('fill=', 'stroke=').replace('fill:', 'stroke:')}{ex}/>"
            )
    else:
        raise ValueError(f"unknown marker shape {shape!r}")
    return f'<path d="{d}" {paint}{ex}/>'


def _thin(pts: list[tuple[float, float]], limit: int) -> list[tuple[float, float]]:
    n = len(pts)
    if n <= limit:
        return pts
    return [pts[round(i * (n - 1) / (limit - 1))] for i in range(limit)]


# --------------------------------------------------------------------------- label geometry

_Box = tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels
_Seg = tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels


def _text_w(s: str, size: float = 11, bold: bool = False) -> float:
    return len(s) * _CHAR_W * size / 11 * (1.06 if bold else 1.0)


def _text_box(
    x: float, y: float, s: str, anchor: str = "start", size: float = 11, bold: bool = False
) -> _Box:
    """Approximate bounding box of an unrotated text element with baseline at ``y``."""
    w = _text_w(s, size, bold)
    x0 = x if anchor == "start" else x - w if anchor == "end" else x - w / 2
    return (x0, y - 0.82 * size, x0 + w, y + 0.25 * size)


def _overlap(a: _Box, b: _Box) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def _outside(box: _Box, bounds: _Box) -> float:
    return (box[2] - box[0]) * (box[3] - box[1]) - _overlap(box, bounds)


def _clip_len(box: _Box, seg: _Seg) -> float:
    """Length of the part of ``seg`` inside ``box`` (Liang-Barsky)."""
    x0, y0, x1, y1 = seg
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - box[0]), (dx, box[2] - x0), (-dy, y0 - box[1]), (dy, box[3] - y0)):
        if p == 0:
            if q < 0:
                return 0.0
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return 0.0
    return (t1 - t0) * math.hypot(dx, dy)


@dataclass
class _Obstacles:
    """Things a label should avoid: boxes (labels, marker shapes, bars), line segments and
    scatter points. :meth:`cost` scores a candidate label box (0 = free)."""

    boxes: list[tuple[_Box, float]]
    segs: list[_Seg]
    pts: list[tuple[float, float]]

    def cost(self, box: _Box) -> float:
        c = sum(w * _overlap(box, b) for b, w in self.boxes)
        x0, y0, x1, y1 = box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2
        c += 25.0 * sum(_clip_len((x0, y0, x1, y1), s) for s in self.segs)
        c += 3.0 * sum(1 for x, y in self.pts if x0 <= x <= x1 and y0 <= y <= y1)
        return c


def _best(cands: Sequence[tuple[float, _Box]], obs: _Obstacles, bounds: _Box) -> int:
    """Index of the cheapest candidate (penalty, box); ties go to the earliest."""
    best_i, best_c = 0, math.inf
    for i, (pen, box) in enumerate(cands):
        c = pen + obs.cost(box) + 200.0 * _outside(box, bounds)
        if c < best_c - 1e-9:
            best_i, best_c = i, c
    return best_i


# --------------------------------------------------------------------------- svg document


def _svg(width: int, height: int, title: str, desc: str | None, body: Sequence[str]) -> str:
    label = title if not desc else f"{title}. {desc}"
    head = (
        f'<svg xmlns="http://www.w3.org/2000/svg" class="sg-chart" viewBox="0 0 {width} {height}"'
        f' width="100%" preserveAspectRatio="xMidYMid meet" role="img"'
        f' aria-label="{_esc(label)}" font-family="{_FONT}" font-size="11"'
        f' style="max-width:{width}px;height:auto">'
        f"<title>{_esc(title)}</title>"
    )
    if desc:
        head += f"<desc>{_esc(desc)}</desc>"
    content = "".join(body)
    if "\x00CLIP\x00" in content:
        digest = hashlib.sha1((head + content).encode("utf-8")).hexdigest()[:10]
        content = content.replace("\x00CLIP\x00", f"sg-clip-{digest}")
    return head + content + "</svg>"


def _placeholder(title: str, width: int, height: int, desc: str | None, msg: str) -> str:
    body = [
        f'<rect x="1" y="1" width="{width - 2}" height="{height - 2}" rx="6" fill="none"'
        f' stroke="currentColor" stroke-opacity="0.25" stroke-dasharray="{_DASH}"/>',
        _text(width / 2, height / 2 + 4, msg, anchor="middle", size=13, extra='fill-opacity="0.7"'),
    ]
    return _svg(width, height, title, desc or msg, body)


def _check_dims(width: int, height: int) -> tuple[int, int]:
    w, h = int(width), int(height)
    if w < 120 or h < 100:
        raise ValueError("chart width must be >= 120 and height >= 100")
    return w, h


def _domain(
    vals: list[float], fixed: tuple[float, float] | None, target: int, *, log: bool
) -> tuple[float, float, list[float]]:
    if log:
        if fixed is not None:
            lo, hi = float(fixed[0]), float(fixed[1])
            if not (lo > 0 and hi > lo):
                raise ValueError("log axis range must satisfy 0 < lo < hi")
        else:
            pos = [v for v in vals if v > 0]
            lo, hi = min(pos), max(pos)
            if hi <= lo * (1 + 1e-12):
                lo, hi = lo / 2, hi * 2
            span = math.log10(hi / lo)
            pad = 10 ** (max(span, 0.3) * 0.05)
            lo, hi = lo / pad, hi * pad
        return lo, hi, log_ticks(lo, hi)
    if fixed is not None:
        lo, hi = float(fixed[0]), float(fixed[1])
        if not hi > lo:
            raise ValueError("axis range must satisfy lo < hi")
        tol = 1e-9 * max(1.0, abs(lo), abs(hi))
        ticks = [t for t in nice_ticks(lo, hi, target) if lo - tol <= t <= hi + tol]
        return lo, hi, ticks
    ticks = nice_ticks(min(vals), max(vals), target)
    return ticks[0], ticks[-1], ticks


def _tick_step(ticks: Sequence[float]) -> float | None:
    return ticks[1] - ticks[0] if len(ticks) >= 2 else None


def _tick_labels(ticks: Sequence[float], fmt: ValueFormat, log: bool) -> list[str]:
    if log:
        return [format_value(t, fmt) for t in ticks]
    step = _tick_step(ticks)
    return [format_value(t, fmt, step=step) for t in ticks]


@dataclass
class _LegendItem:
    name: str
    color: str
    kind: str  # "line" | "shape"
    dashed: bool = False
    shape: str = "circle"


def _legend_swatch(item: _LegendItem, x: float, y: float) -> str:
    if item.kind == "line":
        dash = f' stroke-dasharray="{_DASH}"' if item.dashed else ""
        return (
            f'<line x1="{_f(x)}" y1="{_f(y)}" x2="{_f(x + 18)}" y2="{_f(y)}" stroke-width="2.5"'
            f" {_paint(stroke=item.color)}{dash}/>"
        )
    return _shape(item.shape, x + 9, y, 4.5, _paint(fill=item.color))


def _legend(
    items: Sequence[_LegendItem], pos: str, x0: float, y0: float, avail: float, max_chars: int
) -> tuple[list[str], float]:
    """Render legend entries; returns (elements, extent) where extent is height used."""
    out: list[str] = ['<g class="sg-legend">']
    if pos == "right":
        y = y0 + 6
        for item in items:
            label = _trunc(item.name, max_chars)
            out.append(f"<g><title>{_esc(item.name)}</title>")
            out.append(_legend_swatch(item, x0, y))
            out.append(_text(x0 + 24, y + 4, label))
            out.append("</g>")
            y += 17
        out.append("</g>")
        return out, y - y0
    x, y = x0, y0 + 6
    for item in items:
        label = _trunc(item.name, max_chars)
        w = 24 + len(label) * _CHAR_W + 14
        if x > x0 and x + w > x0 + avail:
            x, y = x0, y + 17
        out.append(f"<g><title>{_esc(item.name)}</title>")
        out.append(_legend_swatch(item, x, y))
        out.append(_text(x + 24, y + 4, label))
        out.append("</g>")
        x += w
    out.append("</g>")
    return out, y - y0 + 11


def _legend_rows(items: Sequence[_LegendItem], avail: float, max_chars: int) -> int:
    rows, x = 1, 0.0
    for item in items:
        w = 24 + len(_trunc(item.name, max_chars)) * _CHAR_W + 14
        if x > 0 and x + w > avail:
            rows, x = rows + 1, 0.0
        x += w
    return rows


def _valid_pts(
    raw: Sequence[tuple[float | None, float | None]], xlog: bool
) -> list[list[tuple[float, float]]]:
    """Split into contiguous runs of valid points."""
    runs: list[list[tuple[float, float]]] = [[]]
    for p in raw:
        try:
            x, y = _num(p[0]), _num(p[1])
        except (TypeError, IndexError, KeyError):
            x = y = None
        if x is None or y is None or (xlog and x <= 0):
            if runs[-1]:
                runs.append([])
            continue
        runs[-1].append((x, y))
    return [r for r in runs if r]


# --------------------------------------------------------------------------- xy charts


def xy_chart(
    series: Series | Sequence[Series] = (),
    *,
    title: str,
    x_label: str = "",
    y_label: str = "",
    width: int = 640,
    height: int = 360,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    x_log: bool = False,
    x_format: ValueFormat = "number",
    y_format: ValueFormat = "number",
    markers: Sequence[Marker] = (),
    error_bars: Sequence[ErrorBar] = (),
    ref_lines: Sequence[RefLine] = (),
    diagonal: Diagonal | bool | None = None,
    legend: LegendPos = "auto",
    default_kind: SeriesKind = "line",
    show_title: bool = False,
    desc: str | None = None,
    empty_message: str = "No data",
) -> str:
    """Render lines, step lines, scatter points, labelled markers, error bars and reference lines
    on shared numeric axes. Returns a standalone ``<svg>`` string."""
    width, height = _check_dims(width, height)
    series_list = [series] if isinstance(series, Series) else list(series)
    if isinstance(diagonal, bool):
        diagonal = Diagonal() if diagonal else None

    # ---- normalise data
    prepared: list[tuple[Series, SeriesKind, str, list[list[tuple[float, float]]]]] = []
    auto = 0
    for s in series_list:
        kind: SeriesKind = s.kind or default_kind
        key = _color_key(s.style, auto)
        if s.style is None:
            auto += 1
        runs = _valid_pts(s.points, x_log)
        if kind == "scatter":
            flat = [p for r in runs for p in r]
            runs = [_thin(flat, MAX_SCATTER_POINTS)] if flat else []
        else:
            runs = [_thin(r, _MAX_LINE_POINTS) for r in runs]
        prepared.append((s, kind, key, runs))

    mk: list[tuple[Marker, float, float]] = []
    for m in markers:
        x, y = _num(m.x), _num(m.y)
        if x is not None and y is not None and not (x_log and x <= 0):
            mk.append((m, x, y))
    eb: list[tuple[ErrorBar, float, float | None, float | None, float | None]] = []
    for e in error_bars:
        x = _num(e.x)
        if x is None or (x_log and x <= 0):
            continue
        y, lo, hi = _num(e.y), _num(e.lo), _num(e.hi)
        if y is None and (lo is None or hi is None):
            continue
        eb.append((e, x, y, lo, hi))
    refs = [
        r
        for r in ref_lines
        if _num(r.value) is not None and not (r.axis == "x" and x_log and r.value <= 0)
    ]

    xs: list[float] = []
    ys: list[float] = []
    for _, _, _, runs in prepared:
        for r in runs:
            xs.extend(p[0] for p in r)
            ys.extend(p[1] for p in r)
    for _, x, y in mk:
        xs.append(x)
        ys.append(y)
    for _, x, y, lo, hi in eb:
        xs.append(x)
        ys.extend(v for v in (y, lo, hi) if v is not None)
    if not xs:
        return _placeholder(title, width, height, desc, empty_message)
    for r in refs:
        (xs if r.axis == "x" else ys).append(float(r.value))

    # ---- legend items
    items: list[_LegendItem] = []
    for s, kind, key, _ in prepared:
        if s.in_legend and s.name:
            if kind == "scatter":
                items.append(_LegendItem(s.name, key, "shape", shape=s.marker))
            else:
                items.append(_LegendItem(s.name, key, "line", dashed=s.dashed))
    for m, _, _ in mk:
        if m.in_legend and m.label:
            items.append(_LegendItem(m.label, _color_key(m.style, 0), "shape", shape=m.shape))
    pos = legend
    if not items:
        pos = "none"
    elif pos == "auto":
        pos = "right" if width >= 560 else "bottom"

    # ---- layout
    top = 12 + (24 if show_title else 0)
    right_w = 0.0
    if pos == "right":
        longest = max(len(_trunc(i.name, 22)) for i in items)
        right_w = min(24 + longest * _CHAR_W + 16, width * 0.32)
    legend_chars = 22 if pos == "right" else 30
    if pos == "right":
        legend_chars = max(4, int((right_w - 40) / _CHAR_W))
    bottom = 22 + (18 if x_label else 0) + 6
    legend_rows = 0
    if pos == "bottom":
        legend_rows = _legend_rows(items, width - 24, legend_chars)
        bottom += legend_rows * 17 + 6
    plot_h_est = height - top - bottom
    y_target = max(2, min(6, int(plot_h_est // 50)))
    ylo, yhi, yticks = _domain(ys, y_range, y_target, log=False)
    ylabels = _tick_labels(yticks, y_format, False)
    tick_w = max((len(t) for t in ylabels), default=1) * _CHAR_W
    left = 8 + (18 if y_label else 0) + tick_w + 8
    px0, px1 = left, width - 14 - right_w
    py0, py1 = top, height - bottom
    if px1 - px0 < 40 or py1 - py0 < 30:
        raise ValueError("chart too small for its labels; increase width/height")
    x_target = max(2, min(7, int((px1 - px0) // 85)))
    xlo, xhi, xticks = _domain(xs, x_range, x_target, log=x_log)
    xlabels = _tick_labels(xticks, x_format, x_log)
    sx = _Scale(xlo, xhi, px0, px1, x_log)
    sy = _Scale(ylo, yhi, py1, py0)

    body: list[str] = []
    if show_title:
        body.append(_text(px0, 18, title, size=13, bold=True))

    # ---- grid, ticks
    g = ['<g class="sg-axes">']
    for t, lab in zip(yticks, ylabels, strict=True):
        yy = sy(t)
        g.append(
            f'<line x1="{_f(px0)}" y1="{_f(yy)}" x2="{_f(px1)}" y2="{_f(yy)}" stroke="currentColor"'
            f' stroke-opacity="0.12"/>'
        )
        g.append(_text(px0 - 6, yy + 4, lab, anchor="end", extra='fill-opacity="0.8"'))
    # thin x labels if they would collide
    max_lab = max((len(lb) for lb in xlabels), default=1) * _CHAR_W + 8
    stride = 1
    while len(xticks) > 1 and (px1 - px0) / max(1, (len(xticks) - 1)) * stride < max_lab:
        stride += 1
    for i, (t, lab) in enumerate(zip(xticks, xlabels, strict=True)):
        xx = sx(t)
        g.append(
            f'<line x1="{_f(xx)}" y1="{_f(py0)}" x2="{_f(xx)}" y2="{_f(py1)}" stroke="currentColor"'
            f' stroke-opacity="0.08"/>'
        )
        if i % stride == 0:
            g.append(_text(xx, py1 + 16, lab, anchor="middle", extra='fill-opacity="0.8"'))
    g.append(
        f'<path d="M{_f(px0)} {_f(py0)}V{_f(py1)}H{_f(px1)}" fill="none" stroke="currentColor"'
        f' stroke-opacity="0.55"/>'
    )
    g.append("</g>")
    body.extend(g)
    if x_label:
        body.append(_text((px0 + px1) / 2, py1 + 36, x_label, anchor="middle", size=12))
    if y_label:
        cy = (py0 + py1) / 2
        body.append(
            _text(
                14,
                cy,
                y_label,
                anchor="middle",
                size=12,
                extra=f'transform="rotate(-90 14 {_f(cy)})"',
            )
        )

    body.append(
        f'<clipPath id="\x00CLIP\x00"><rect x="{_f(px0 - 6)}" y="{_f(py0 - 6)}"'
        f' width="{_f(px1 - px0 + 12)}" height="{_f(py1 - py0 + 12)}"/></clipPath>'
    )

    # ---- obstacles for label placement (series lines, points, error bars, markers)
    obs = _Obstacles([], [], [])
    for s, kind, _, runs in prepared:
        if kind == "scatter":
            obs.pts.extend((sx(x), sy(y)) for r in runs for x, y in r)
            continue
        for run in runs:
            prev = (sx(run[0][0]), sy(run[0][1]))
            if s.show_points or len(run) == 1:
                obs.pts.append(prev)
            for x, y in run[1:]:
                cur = (sx(x), sy(y))
                if kind == "step":
                    obs.segs.append((prev[0], prev[1], cur[0], prev[1]))
                    obs.segs.append((cur[0], prev[1], cur[0], cur[1]))
                else:
                    obs.segs.append((prev[0], prev[1], cur[0], cur[1]))
                if s.show_points:
                    obs.pts.append(cur)
                prev = cur
    for _, x, y, lo, hi in eb:
        if lo is not None and hi is not None:
            a, b = sy(max(min(lo, hi), ylo)), sy(min(max(lo, hi), yhi))
            obs.segs.append((sx(x), a, sx(x), b))
        if y is not None:
            obs.pts.append((sx(x), sy(y)))
    for m, x, y in mk:
        rr = _marker_extent(m)
        cx, cy = sx(x), sy(y)
        obs.boxes.append(((cx - rr, cy - rr, cx + rr, cy + rr), 50.0))
    bounds = (px0 + 1, py0 - 2, px1 + 8, py1 - 1)

    # ---- reference lines and diagonal (beneath data)
    ref_texts: list[str] = []
    for r in refs:
        v = float(r.value)
        key = _color_key(r.style, 0)
        dash = f' stroke-dasharray="{_DASH}"' if r.dashed else ""
        if r.axis == "y":
            if not sy.contains(v):
                continue
            yy = sy(v)
            body.append(
                f'<line x1="{_f(px0)}" y1="{_f(yy)}" x2="{_f(px1)}" y2="{_f(yy)}"'
                f' stroke-width="1.5" {_paint(stroke=key)}{dash}/>'
            )
            opts = [
                (px1 - 4, yy - 5, "end"),
                (px0 + 4, yy - 5, "start"),
                (px1 - 4, yy + 13, "end"),
                (px0 + 4, yy + 13, "start"),
            ]
        else:
            if not sx.contains(v):
                continue
            xx = sx(v)
            body.append(
                f'<line x1="{_f(xx)}" y1="{_f(py0)}" x2="{_f(xx)}" y2="{_f(py1)}"'
                f' stroke-width="1.5" {_paint(stroke=key)}{dash}/>'
            )
            near_left = xx < (px0 + px1) / 2
            sides = [(xx + 5, "start"), (xx - 5, "end")]
            if not near_left:
                sides.reverse()
            opts = [(x_, y_, a_) for y_ in (py0 + 12, py1 - 6) for x_, a_ in sides]
        if r.label:
            cands = [
                (0.5 * i, _text_box(x_, y_, r.label, a_, 10.5))
                for i, (x_, y_, a_) in enumerate(opts)
            ]
            i = _best(cands, obs, bounds)
            x_, y_, a_ = opts[i]
            ref_texts.append(_text(x_, y_, r.label, anchor=a_, halo=True, size=10.5))
            obs.boxes.append((cands[i][1], 50.0))
    if diagonal is not None and not x_log:
        d0, d1 = max(xlo, ylo), min(xhi, yhi)
        if d1 > d0:
            dash = f' stroke-dasharray="{_DASH}"' if diagonal.dashed else ""
            seg = (sx(d0), sy(d0), sx(d1), sy(d1))
            body.append(
                f'<line x1="{_f(seg[0])}" y1="{_f(seg[1])}" x2="{_f(seg[2])}" y2="{_f(seg[3])}"'
                f' stroke="currentColor" stroke-opacity="0.45" stroke-width="1.25"{dash}/>'
            )
            if diagonal.label:
                lx, ly, ang, box = _diagonal_label(seg, diagonal.label, obs, bounds)
                rot = f"rotate({_f(ang)} {_f(lx)} {_f(ly)})"
                ref_texts.append(
                    _text(
                        lx,
                        ly,
                        diagonal.label,
                        anchor="end",
                        size=10.5,
                        halo=True,
                        extra=f'fill-opacity="0.75" transform="{rot}"',
                    )
                )
                obs.boxes.append((box, 50.0))
            obs.segs.append(seg)
    body.extend(ref_texts)  # above all reference lines, beneath the data

    # ---- series
    body.append('<g clip-path="url(#\x00CLIP\x00)">')
    for s, kind, key, runs in prepared:
        if not runs:
            continue
        body.append(f'<g class="sg-series"><title>{_esc(s.name)}</title>')
        if kind == "scatter":
            pts = runs[0]
            r = s.size if s.size is not None else (2.5 if len(pts) > 300 else 3.5)
            op = s.opacity if s.opacity is not None else (0.55 if len(pts) > 300 else 0.85)
            body.append(f'<g {_paint(fill=key)} fill-opacity="{_f(op)}">')
            for x, y in pts:
                if s.marker == "circle":
                    body.append(f'<circle cx="{_f(sx(x))}" cy="{_f(sy(y))}" r="{_f(r)}"/>')
                else:
                    body.append(_shape(s.marker, sx(x), sy(y), r, ""))
            body.append("</g>")
        else:
            segs = []
            for run in runs:
                d = f"M{_f(sx(run[0][0]))} {_f(sy(run[0][1]))}"
                for x1, y1 in run[1:]:
                    if kind == "step":
                        d += f"H{_f(sx(x1))}V{_f(sy(y1))}"
                    else:
                        d += f"L{_f(sx(x1))} {_f(sy(y1))}"
                segs.append(d)
            dash = f' stroke-dasharray="{_DASH}"' if s.dashed else ""
            op = f' stroke-opacity="{_f(s.opacity)}"' if s.opacity is not None else ""
            body.append(
                f'<path d="{"".join(segs)}" fill="none" stroke-width="{_f(s.width)}"'
                f' stroke-linejoin="round" stroke-linecap="round" {_paint(stroke=key)}{dash}{op}/>'
            )
            if s.show_points or all(len(r) == 1 for r in runs):
                r = s.size if s.size is not None else 3.0
                body.append(f"<g {_paint(fill=key)}>")
                for run in runs:
                    for x, y in run:
                        body.append(_shape(s.marker, sx(x), sy(y), r, ""))
                body.append("</g>")
        body.append("</g>")
    body.append("</g>")

    # ---- error bars
    if eb:
        body.append('<g class="sg-errorbars">')
        for e, x, y, lo, hi in eb:
            if not sx.contains(x):
                continue
            key = _color_key(e.style, 0)
            xx = sx(x)
            if lo is not None and hi is not None:
                a, b = sy(max(min(lo, hi), ylo)), sy(min(max(lo, hi), yhi))
                body.append(
                    f'<path d="M{_f(xx)} {_f(a)}V{_f(b)}M{_f(xx - 4)} {_f(a)}H{_f(xx + 4)}'
                    f'M{_f(xx - 4)} {_f(b)}H{_f(xx + 4)}" fill="none" stroke-width="1.5"'
                    f" {_paint(stroke=key)}/>"
                )
            if e.show_point and y is not None and sy.contains(y):
                body.append(f'<circle cx="{_f(xx)}" cy="{_f(sy(y))}" r="3.5" {_paint(fill=key)}/>')
        body.append("</g>")

    # ---- markers with nudged labels
    if mk:
        body.extend(_draw_markers(mk, sx, sy, obs, bounds))

    # ---- legend
    if pos == "right":
        els, _ = _legend(items, "right", px1 + 16, py0, right_w, legend_chars)
        body.extend(els)
    elif pos == "bottom":
        ly = height - legend_rows * 17 - 6
        els, _ = _legend(items, "bottom", 12, ly, width - 24, legend_chars)
        body.extend(els)
    return _svg(width, height, title, desc, body)


_DIAG_ASC, _DIAG_DESC = 8.6, 2.6  # glyph extent above/below the baseline at 10.5px


def _diagonal_label(
    seg: _Seg, label: str, obs: _Obstacles, bounds: _Box
) -> tuple[float, float, float, _Box]:
    """Place the diagonal's label rotated along the line, offset to one side so it never
    touches it. Returns (x, y, angle in degrees, axis-aligned bbox) for an end-anchored text."""
    x0, y0, x1, y1 = seg
    length = math.hypot(x1 - x0, y1 - y0)
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    nx, ny = -uy, ux  # unit normal; for a rising line this points down-right
    ang = math.degrees(math.atan2(uy, ux))
    w = _text_w(label, 10.5)
    gap = 4.0

    def frame(side: int, pad: float) -> tuple[float, float, list[tuple[float, float]]]:
        # baseline distance from the line so the glyphs keep ``gap`` px clear of it
        off = side * (gap + _DIAG_ASC) if side > 0 else side * (gap + _DIAG_DESC)
        ex, ey = x1 - pad * ux + off * nx, y1 - pad * uy + off * ny
        corners = [
            (ex - a * ux + b * nx, ey - a * uy + b * ny)
            for a in (0.0, w)
            for b in (-_DIAG_ASC, _DIAG_DESC)
        ]
        return ex, ey, corners

    def local_cost(ex: float, ey: float) -> float:
        # obstacles in the label's own (along-line, normal) frame, where it is axis-aligned
        def to_local(px: float, py: float) -> tuple[float, float]:
            dx, dy = px - ex, py - ey
            return -(dx * ux + dy * uy), dx * nx + dy * ny

        box = (-3.0, -_DIAG_ASC - 3.0, w + 3.0, _DIAG_DESC + 3.0)  # with clearance
        c = 0.0
        for s in obs.segs:
            a, b = to_local(s[0], s[1]), to_local(s[2], s[3])
            c += 25.0 * _clip_len(box, (a[0], a[1], b[0], b[1]))
        for p in obs.pts:
            lx, ly = to_local(*p)
            if -2 <= lx <= w + 2 and -_DIAG_ASC - 2 <= ly <= _DIAG_DESC + 2:
                c += 3.0
        return c

    best: tuple[float, float, float, list[tuple[float, float]]] | None = None
    for side in (1, -1):  # below-right of the line first, then above-left
        pad = 6.0
        while pad + w <= length or pad == 6.0:
            ex, ey, corners = frame(side, pad)
            bb = (
                min(p[0] for p in corners),
                min(p[1] for p in corners),
                max(p[0] for p in corners),
                max(p[1] for p in corners),
            )
            out = sum(
                max(0.0, bounds[0] - px, px - bounds[2], bounds[1] - py, py - bounds[3])
                for px, py in corners
            )
            c = 400.0 * out + local_cost(ex, ey) + sum(wt * _overlap(bb, b) for b, wt in obs.boxes)
            c += 0.05 * pad + (0 if side > 0 else 2.0)
            if best is None or c < best[0] - 1e-9:
                best = (c, ex, ey, corners)
            pad += 4.0
    assert best is not None
    _, ex, ey, corners = best
    bb = (
        min(p[0] for p in corners),
        min(p[1] for p in corners),
        max(p[0] for p in corners),
        max(p[1] for p in corners),
    )
    return ex, ey, ang, bb


def _marker_extent(m: Marker) -> float:
    """Half-size of a marker's drawn footprint (shape plus emphasis ring)."""
    r = 6.5 if m.emphasis else 5.0
    return r + 6.0 if m.emphasis else r * 1.45 + 1.0


def _draw_markers(
    mk: Sequence[tuple[Marker, float, float]],
    sx: _Scale,
    sy: _Scale,
    obs: _Obstacles,
    bounds: _Box,
) -> list[str]:
    out = ['<g class="sg-markers">']
    labels: list[tuple[float, float, float, str, bool]] = []  # px, py, extent, text, bold
    shapes: list[str] = []
    for m, x, y in mk:
        if not (sx.contains(x) and sy.contains(y)):
            continue
        cx, cy = sx(x), sy(y)
        key = _color_key(m.style, 0)
        r = 6.5 if m.emphasis else 5.0
        title = f"<title>{_esc(m.label)}</title>" if m.label else ""
        ring = ""
        if m.emphasis:
            ring = (
                f'<circle cx="{_f(cx)}" cy="{_f(cy)}" r="{_f(r + 5)}" fill="none"'
                f' stroke-width="1.5" {_paint(stroke=key)}/>'
            )
        paint = _paint(fill=key)
        if m.shape != "x":
            # filled shapes get a thin background-coloured outline to separate them from lines
            hx = _FALLBACK[key]
            paint = (
                f'fill="{hx}" stroke="#FFFFFF" stroke-width="1.5"'
                f' style="fill:var(--sg-{key},{hx});stroke:var(--sg-bg,#FFFFFF)"'
            )
        shapes.append(f"<g>{title}{ring}{_shape(m.shape, cx, cy, r, paint)}</g>")
        if m.label:
            labels.append((cx, cy, _marker_extent(m), m.label, m.emphasis))
    out.extend(shapes)

    # try positions around each marker (right, left, above, below, diagonals, then rows
    # stacked further away with a leader line) and keep the one with the least overlap
    # against already placed labels, markers, series lines and points; deterministic
    line_h = 13.0
    for cx, cy, rr, text, bold in sorted(labels, key=lambda t: (t[1], t[0])):
        g = rr + 3.0
        d = rr * 0.72 + 2.0
        opts: list[tuple[float, float, float, str, bool]] = [  # penalty, x, y, anchor, leader
            (0.0, cx + g, cy + 4, "start", False),
            (1.0, cx - g, cy + 4, "end", False),
            (2.0, cx, cy - rr - 5, "middle", False),
            (3.0, cx, cy + rr + 11, "middle", False),
            (4.0, cx + d, cy - d - 3, "start", False),
            (4.5, cx + d, cy + d + 9, "start", False),
            (5.0, cx - d, cy - d - 3, "end", False),
            (5.5, cx - d, cy + d + 9, "end", False),
        ]
        for k in range(1, 7):
            for sgn in (1, -1):
                for dx, anchor in ((g, "start"), (-g, "end")):
                    pen = 14.0 * k + (0.5 if sgn < 0 else 0) + (0.25 if dx < 0 else 0)
                    opts.append((pen, cx + dx, cy + 4 + sgn * k * line_h, anchor, True))
        cands = [(p, _text_box(x, y, text, a, bold=bold)) for p, x, y, a, _ in opts]
        i = _best(cands, obs, bounds)
        _, lx, ly, anchor, leader = opts[i]
        box = cands[i][1]
        obs.boxes.append((box, 50.0))
        if leader:
            # from the marker's edge to the nearest point of the label box
            tx = min(max(cx, box[0] - 2), box[2] + 2)
            ty = min(max(cy, box[1]), box[3])
            dist = math.hypot(tx - cx, ty - cy)
            if dist > rr + 2:
                sx0 = cx + (tx - cx) * (rr + 1) / dist
                sy0 = cy + (ty - cy) * (rr + 1) / dist
                out.append(
                    f'<line x1="{_f(sx0)}" y1="{_f(sy0)}" x2="{_f(tx)}" y2="{_f(ty)}"'
                    f' stroke="currentColor" stroke-opacity="0.4"/>'
                )
        out.append(_text(lx, ly, text, anchor=anchor, bold=bold, halo=True))
    out.append("</g>")
    return out


def line_chart(series: Series | Sequence[Series], **kwargs: object) -> str:
    """:func:`xy_chart` with series drawn as lines unless their ``kind`` says otherwise."""
    kwargs.setdefault("default_kind", "line")
    return xy_chart(series, **kwargs)  # type: ignore[arg-type]


def scatter(series: Series | Sequence[Series], **kwargs: object) -> str:
    """:func:`xy_chart` with series drawn as points unless their ``kind`` says otherwise.
    Each scatter series is thinned to at most :data:`MAX_SCATTER_POINTS` points."""
    kwargs.setdefault("default_kind", "scatter")
    return xy_chart(series, **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- bar chart


def bar_with_ci(
    categories: Sequence[str],
    values: Sequence[float | None],
    lows: Sequence[float | None] | None = None,
    highs: Sequence[float | None] | None = None,
    *,
    title: str,
    x_label: str = "",
    y_label: str = "",
    y_format: ValueFormat = "number",
    y_range: tuple[float, float] | None = None,
    ref_line: RefLine | None = None,
    ref_lines: Sequence[RefLine] = (),
    annotations: Sequence[str | None] | None = None,
    styles: Sequence[Style] | None = None,
    style: Style = 1,
    width: int = 640,
    height: int = 320,
    show_title: bool = False,
    desc: str | None = None,
    empty_message: str = "No data",
) -> str:
    """Vertical bars per category with optional ``[low, high]`` whiskers, per-bar annotations
    (e.g. ``"n=42"``) and horizontal reference lines (e.g. a tolerance)."""
    width, height = _check_dims(width, height)
    n = len(categories)
    for name, seq in (
        ("values", values),
        ("lows", lows),
        ("highs", highs),
        ("annotations", annotations),
        ("styles", styles),
    ):
        if seq is not None and len(seq) != n:
            raise ValueError(f"{name} has {len(seq)} items, expected {n}")
    vals = [_num(v) for v in values]
    los = [_num(v) for v in lows] if lows is not None else [None] * n
    his = [_num(v) for v in highs] if highs is not None else [None] * n
    refs = [
        r
        for r in ([ref_line] if ref_line else []) + list(ref_lines)
        if r.axis == "y" and _num(r.value) is not None
    ]
    if (
        n == 0
        or all(v is None for v in vals)
        and all(lo is None or hi is None for lo, hi in zip(los, his, strict=True))
    ):
        return _placeholder(title, width, height, desc, empty_message)

    ys = [0.0] + [v for v in vals if v is not None]
    ys += [v for v in los if v is not None] + [v for v in his if v is not None]
    ys += [float(r.value) for r in refs]

    top = 14 + (24 if show_title else 0) + (12 if annotations else 0)
    bottom = 24 + (18 if x_label else 0) + 6
    y_target = max(2, min(6, int((height - top - bottom) // 50)))
    ylo, yhi, yticks = _domain(ys, y_range, y_target, log=False)
    ylabels = _tick_labels(yticks, y_format, False)
    tick_w = max((len(t) for t in ylabels), default=1) * _CHAR_W
    left = 8 + (18 if y_label else 0) + tick_w + 8
    px0, px1 = left, width - 14
    py0, py1 = top, height - bottom
    sy = _Scale(ylo, yhi, py1, py0)
    band = (px1 - px0) / n
    bw = min(band * 0.62, 64.0)

    body: list[str] = []
    if show_title:
        body.append(_text(px0, 18, title, size=13, bold=True))
    body.append('<g class="sg-axes">')
    for t, lab in zip(yticks, ylabels, strict=True):
        yy = sy(t)
        body.append(
            f'<line x1="{_f(px0)}" y1="{_f(yy)}" x2="{_f(px1)}" y2="{_f(yy)}" stroke="currentColor"'
            f' stroke-opacity="0.12"/>'
        )
        body.append(_text(px0 - 6, yy + 4, lab, anchor="end", extra='fill-opacity="0.8"'))
    body.append("</g>")

    def clamp(v: float) -> float:
        return min(max(v, ylo), yhi)

    base = sy(clamp(0.0))
    label_stride = 1 if band >= 28 else math.ceil(28 / band)
    max_chars = max(3, int(band * label_stride / _CHAR_W) - 1)

    # ---- geometry pass: bars, whiskers, annotations; then reference-line labels in a free
    # spot; annotations that still collide with a reference label are nudged clear of it
    obs = _Obstacles([], [], [])
    ann_pos: list[tuple[float, float] | None] = []
    for i in range(n):
        cx = px0 + band * (i + 0.5)
        v, lo, hi = vals[i], los[i], his[i]
        top_y = base
        if v is not None:
            vy = sy(clamp(v))
            obs.boxes.append(((cx - bw / 2, min(vy, base), cx + bw / 2, max(vy, base)), 20.0))
            top_y = min(top_y, vy)
        if lo is not None and hi is not None:
            a, b = sy(clamp(min(lo, hi))), sy(clamp(max(lo, hi)))
            cap = min(bw * 0.3, 8)
            obs.boxes.append(((cx - cap, b - 1, cx + cap, a + 1), 20.0))
            top_y = min(top_y, b)
        if annotations is not None and annotations[i]:
            ann_pos.append((cx, top_y - 5))
            obs.boxes.append((_text_box(cx, top_y - 5, str(annotations[i]), "middle", 10), 50.0))
        else:
            ann_pos.append(None)
    bounds = (px0 + 1, 2.0, px1, py1 - 1)
    ref_draw: list[tuple[RefLine, float, str, str, float, float, str]] = []
    ref_boxes: list[_Box] = []
    for r in refs:
        v = float(r.value)
        if not sy.contains(v):
            continue
        yy = sy(v)
        key = _color_key(r.style, 0)
        dash = f' stroke-dasharray="{_DASH}"' if r.dashed else ""
        lx, ly, anchor = px1 - 4, yy - 5, "end"
        if r.label:
            opts = [
                (px1 - 4, yy - 5, "end"),
                (px0 + 4, yy - 5, "start"),
                (px1 - 4, yy + 13, "end"),
                (px0 + 4, yy + 13, "start"),
            ]
            for i in range(n - 1, -1, -1):  # band centres, right to left
                cx = px0 + band * (i + 0.5)
                opts += [(cx, yy - 5, "middle"), (cx, yy + 13, "middle")]
            cands = [
                (0.5 * j, _text_box(x_, y_, r.label, a_, 10.5))
                for j, (x_, y_, a_) in enumerate(opts)
            ]
            j = _best(cands, obs, bounds)
            lx, ly, anchor = opts[j]
            ref_boxes.append(cands[j][1])
            obs.boxes.append((cands[j][1], 50.0))
        ref_draw.append((r, yy, key, dash, lx, ly, anchor))
    if annotations is not None:
        for i, pos in enumerate(ann_pos):
            if pos is None:
                continue
            cx, ay = pos
            text = str(annotations[i])
            for _ in range(len(ref_boxes) + 1):
                hit = next(
                    (b for b in ref_boxes if _overlap(_text_box(cx, ay, text, "middle", 10), b)),
                    None,
                )
                if hit is None:
                    break
                up = hit[1] - 1 - 0.25 * 10
                ay = up if up - 0.82 * 10 >= 2 else hit[3] + 1 + 0.82 * 10
            ann_pos[i] = (cx, ay)

    body.append('<g class="sg-bars">')
    for i, cat in enumerate(categories):
        cx = px0 + band * (i + 0.5)
        key = _color_key(styles[i] if styles is not None else style, i)
        v, lo, hi = vals[i], los[i], his[i]
        tip = [str(cat)]
        if v is not None:
            tip.append(format_value(v, y_format))
        if lo is not None and hi is not None:
            tip.append(f"[{format_value(lo, y_format)}, {format_value(hi, y_format)}]")
        if annotations is not None and annotations[i]:
            tip.append(str(annotations[i]))
        body.append(f"<g><title>{_esc(' '.join(tip))}</title>")
        if v is not None:
            vy = sy(clamp(v))
            y0, h = min(vy, base), abs(base - vy)
            body.append(
                f'<rect x="{_f(cx - bw / 2)}" y="{_f(y0)}" width="{_f(bw)}" height="{_f(h)}"'
                f' rx="2" {_paint(fill=key)}/>'
            )
        if lo is not None and hi is not None:
            a, b = sy(clamp(min(lo, hi))), sy(clamp(max(lo, hi)))
            cap = min(bw * 0.3, 8)
            body.append(
                f'<path d="M{_f(cx)} {_f(a)}V{_f(b)}M{_f(cx - cap)} {_f(a)}H{_f(cx + cap)}'
                f'M{_f(cx - cap)} {_f(b)}H{_f(cx + cap)}" fill="none" stroke="currentColor"'
                f' stroke-width="1.5" stroke-opacity="0.85"/>'
            )
        pos = ann_pos[i]
        if annotations is not None and annotations[i] and pos is not None:
            body.append(
                _text(
                    cx,
                    pos[1],
                    str(annotations[i]),
                    anchor="middle",
                    size=10,
                    extra='fill-opacity="0.8"',
                )
            )
        body.append("</g>")
        if i % label_stride == 0:
            full = str(cat)
            short = _trunc(full, max_chars)
            body.append(f"<g><title>{_esc(full)}</title>")
            body.append(_text(cx, py1 + 16, short, anchor="middle", extra='fill-opacity="0.8"'))
            body.append("</g>")
    body.append("</g>")
    body.append(
        f'<path d="M{_f(px0)} {_f(py0)}V{_f(py1)}H{_f(px1)}" fill="none" stroke="currentColor"'
        f' stroke-opacity="0.55"/>'
    )
    if ylo < 0 < yhi:
        body.append(
            f'<line x1="{_f(px0)}" y1="{_f(base)}" x2="{_f(px1)}" y2="{_f(base)}"'
            f' stroke="currentColor" stroke-opacity="0.55"/>'
        )
    for r, yy, key, dash, lx, ly, anchor in ref_draw:
        body.append(
            f'<line x1="{_f(px0)}" y1="{_f(yy)}" x2="{_f(px1)}" y2="{_f(yy)}" stroke-width="1.5"'
            f" {_paint(stroke=key)}{dash}/>"
        )
        if r.label:
            body.append(_text(lx, ly, r.label, anchor=anchor, halo=True, size=10.5))
    if x_label:
        body.append(_text((px0 + px1) / 2, py1 + 36, x_label, anchor="middle", size=12))
    if y_label:
        cy = (py0 + py1) / 2
        body.append(
            _text(
                14,
                cy,
                y_label,
                anchor="middle",
                size=12,
                extra=f'transform="rotate(-90 14 {_f(cy)})"',
            )
        )
    return _svg(width, height, title, desc, body)
