from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import pytest

from shadowgate.svg import (
    CHART_CSS,
    MAX_SCATTER_POINTS,
    Diagonal,
    ErrorBar,
    Marker,
    RefLine,
    Series,
    _clip_len,
    _overlap,
    _text_box,
    _text_w,
    bar_with_ci,
    format_value,
    line_chart,
    log_ticks,
    nice_ticks,
    scatter,
    xy_chart,
)

NS = "{http://www.w3.org/2000/svg}"


def parse(svg: str) -> ET.Element:
    root = ET.fromstring(svg)
    assert root.tag == f"{NS}svg"
    assert root.get("role") == "img"
    assert root.get("viewBox", "").startswith("0 0 ")
    assert root.get("width") == "100%"
    assert root.get("preserveAspectRatio") == "xMidYMid meet"
    assert root.get("aria-label")
    title = root.find(f"{NS}title")
    assert title is not None and title.text
    assert "nan" not in svg.lower().replace("nanos", "")
    assert "inf" not in svg.replace("info", "")
    return root


def texts(root: ET.Element) -> list[str]:
    return ["".join(t.itertext()) for t in root.iter(f"{NS}text")]


def pareto_svg(**kw: object) -> str:
    cloud = [(0.001 + i * 0.00002, 0.6 + (i % 17) / 60) for i in range(200)]
    frontier = [(0.001, 0.62), (0.002, 0.8), (0.004, 0.88), (0.008, 0.9)]
    return scatter(
        [
            Series("operating points", cloud, style="muted"),
            Series("Pareto frontier", frontier, kind="line", style=1),
        ],
        title="Accuracy vs cost",
        x_label="cost per task",
        y_label="accuracy",
        x_format="currency",
        y_format="percent",
        markers=[
            Marker(0.001, 0.62, "only:fast", shape="square"),
            Marker(0.008, 0.9, "only:slow", shape="triangle"),
            Marker(0.009, 0.95, "oracle", shape="diamond"),
            Marker(0.004, 0.88, "recommended", shape="star", style="accent", emphasis=True),
        ],
        **kw,
    )


# ---------------------------------------------------------------- ticks & formatting


def test_nice_ticks_basic() -> None:
    assert nice_ticks(0, 1) == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert nice_ticks(0, 100, 5) == [0, 20, 40, 60, 80, 100]
    t = nice_ticks(0.13, 0.87, 5)
    assert t[0] <= 0.13 and t[-1] >= 0.87
    steps = {round(b - a, 12) for a, b in zip(t, t[1:], strict=False)}
    assert len(steps) == 1
    step = steps.pop()
    mant = step / 10 ** math.floor(math.log10(step))
    assert round(mant, 9) in (1, 2, 5)


def test_nice_ticks_edge_cases() -> None:
    assert nice_ticks(5, 5)[0] < 5 < nice_ticks(5, 5)[-1]
    assert nice_ticks(0, 0) == nice_ticks(-1, 1)
    assert nice_ticks(1, 0) == nice_ticks(0, 1)
    assert nice_ticks(-3, 7)[0] <= -3
    assert all(abs(v - round(v, 6)) < 1e-12 for v in nice_ticks(0.0001, 0.0009))
    with pytest.raises(ValueError):
        nice_ticks(float("nan"), 1)


def test_log_ticks() -> None:
    assert log_ticks(1e-4, 1e-1) == [1e-4, 1e-3, 1e-2, 1e-1]
    assert log_ticks(1, 1e12, max_ticks=7) == [10.0**k for k in range(0, 13, 2)]
    few = log_ticks(1, 30)
    assert few == [1, 2, 5, 10, 20]
    narrow = log_ticks(0.0011, 0.0019)
    assert len(narrow) >= 2 and all(0.0011 <= t <= 0.0019 for t in narrow)
    with pytest.raises(ValueError):
        log_ticks(0, 1)


def test_format_value() -> None:
    assert format_value(0.0012, "currency") == "$0.0012"
    assert format_value(0.035, "currency") == "$0.035"
    assert format_value(0.5, "currency") == "$0.50"
    assert format_value(1234.5, "currency") == "$1,234.50"
    assert format_value(0, "currency") == "$0"
    assert format_value(-0.01, "currency") == "-$0.010"
    assert format_value(0.0005, "currency", step=0.0005) == "$0.0005"
    assert format_value(1, "currency", step=0.5) == "$1.00"
    assert format_value(0.25, "percent") == "25%"
    assert format_value(0.012, "percent") == "1.2%"
    assert format_value(0.05, "percent", step=0.05) == "5%"
    assert format_value(0.125, "percent", step=0.025) == "12.5%"
    assert format_value(0.2, step=0.2) == "0.2"
    assert format_value(-1e-17, step=0.2) == "0.0"
    assert format_value(12345.0) == "12,345"
    assert format_value(0.00123456) == "0.00123"
    assert format_value(float("nan")) == "n/a"
    assert format_value(None) == "n/a"  # type: ignore[arg-type]


# ---------------------------------------------------------------- charts


def test_pareto_scatter_well_formed() -> None:
    svg = pareto_svg()
    root = parse(svg)
    labels = texts(root)
    for name in ("only:fast", "only:slow", "oracle", "recommended", "Pareto frontier"):
        assert name in labels
    assert "var(--sg-series-1,#0072B2)" in svg
    assert "currentColor" in svg
    assert any(lb.startswith("$") for lb in labels)
    assert any(lb.endswith("%") for lb in labels)


def test_log_x_chart() -> None:
    svg = pareto_svg(x_log=True)
    root = parse(svg)
    assert any(lb.startswith("$0.00") for lb in texts(root))
    # non-positive x values are skipped on a log axis
    svg2 = line_chart(Series("s", [(0, 1), (-1, 2), (1, 3), (10, 4)]), title="t", x_log=True)
    parse(svg2)


def test_reliability_diagram() -> None:
    bins = [(0.55, 0.5, 0.4, 0.6), (0.75, 0.7, 0.62, 0.78), (0.95, 0.97, 0.93, 0.99)]
    svg = xy_chart(
        [Series("observed", [(b[0], b[1]) for b in bins], show_points=True)],
        title="Reliability",
        x_label="confidence",
        y_label="accuracy",
        x_range=(0, 1),
        y_range=(0, 1),
        x_format="percent",
        y_format="percent",
        error_bars=[ErrorBar(*b) for b in bins],
        diagonal=Diagonal("perfect calibration"),
    )
    root = parse(svg)
    assert "perfect calibration" in texts(root)
    assert 'stroke-dasharray="6 4"' in svg


def test_risk_coverage_step() -> None:
    pts = [(i / 20, 0.02 + (i / 20) ** 2 * 0.2) for i in range(1, 21)]
    svg = line_chart(
        Series("risk", pts, kind="step"),
        title="Risk-coverage",
        y_format="percent",
        x_format="percent",
        ref_lines=[RefLine("y", 0.05, "tolerance 5%")],
        legend="none",
    )
    root = parse(svg)
    assert "tolerance 5%" in texts(root)
    assert "H" in svg and "V" in svg


def test_bar_with_ci() -> None:
    svg = bar_with_ci(
        ["0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"],
        [0.12, 0.08, None, 0.03, 0.01],
        [0.05, 0.03, None, 0.01, 0.0],
        [0.25, 0.15, None, 0.07, 0.03],
        title="Skipped-case disagreement by confidence bin",
        y_format="percent",
        ref_line=RefLine("y", 0.05, "tolerance"),
        annotations=["n=20", "n=31", None, "n=60", "n=200"],
        styles=["accent", "accent", None, 1, 1],
    )
    root = parse(svg)
    labels = texts(root)
    assert "tolerance" in labels and "n=200" in labels
    assert len(list(root.iter(f"{NS}rect"))) == 4


def test_bar_many_long_categories() -> None:
    cats = [f"category number {i} with a long name" for i in range(40)]
    svg = bar_with_ci(cats, [i / 40 for i in range(40)], title="many", width=320)
    root = parse(svg)
    assert any(t.endswith("…") for t in texts(root))
    titles = ["".join(t.itertext()) for t in root.iter(f"{NS}title")]
    assert cats[0] in titles


def test_bar_length_mismatch() -> None:
    with pytest.raises(ValueError):
        bar_with_ci(["a", "b"], [1.0], title="x")


@pytest.mark.parametrize(
    "svg",
    [
        line_chart([], title="empty"),
        line_chart(Series("s", []), title="empty series"),
        scatter(
            Series("s", [(None, 1), (float("nan"), 2), (1, float("inf"))]), title="invalid values"
        ),
        bar_with_ci([], [], title="empty bars"),
        bar_with_ci(["a"], [None], title="none bar"),
    ],
)
def test_empty_placeholder(svg: str) -> None:
    root = parse(svg)
    assert "No data" in texts(root)


def test_single_point_and_flat_values() -> None:
    parse(line_chart(Series("one", [(3, 3)]), title="single"))
    parse(line_chart(Series("flat", [(1, 0.5), (2, 0.5), (3, 0.5)]), title="flat"))
    parse(line_chart(Series("zero", [(0, 0), (0, 0)]), title="zeros"))
    parse(scatter(Series("one", [(0.002, 0.9)]), title="single", x_log=True))
    parse(bar_with_ci(["a"], [0.0], title="zero bar"))


def test_nan_gaps_break_lines() -> None:
    svg = line_chart(
        Series("gappy", [(0, 1), (1, None), (2, 3), (3, float("nan")), (4, 5), (5, 6)]),
        title="gaps",
    )
    parse(svg)
    path = next(p for p in ET.fromstring(svg).iter(f"{NS}path") if "stroke-linejoin" in p.attrib)
    assert path.get("d", "").count("M") == 3


def test_scatter_thinning() -> None:
    pts = [(i * 0.001, (i % 97) / 97) for i in range(10_000)]
    svg = scatter(Series("many", pts), title="many points")
    root = parse(svg)
    assert len(list(root.iter(f"{NS}circle"))) <= MAX_SCATTER_POINTS + 5


def test_escaping() -> None:
    nasty = "a<b & c>\"d'\x01"
    parse(pareto_svg(desc=nasty))
    svg = xy_chart(
        [Series(nasty, [(0, 0), (1, 1)])],
        title=nasty,
        x_label=nasty,
        y_label=nasty,
        markers=[Marker(0.5, 0.5, nasty)],
        ref_lines=[RefLine("y", 0.3, nasty)],
        desc=nasty,
    )
    root = parse(svg)
    assert "a<b & c>\"d'" in texts(root)
    assert root.find(f"{NS}title").text == "a<b & c>\"d'"  # type: ignore[union-attr]
    bars = bar_with_ci([nasty], [1.0], title=nasty, annotations=[nasty])
    parse(bars)


def test_legend_truncation_keeps_full_name() -> None:
    long = "a very long series name that will not fit in the legend column at all"
    svg = line_chart([Series(long, [(0, 0), (1, 1)])], title="t")
    root = parse(svg)
    assert long not in texts(root)
    assert long in ["".join(t.itertext()) for t in root.iter(f"{NS}title")]
    # bottom legend for narrow charts
    parse(line_chart([Series(long, [(0, 0), (1, 1)]), Series("b", [(0, 1)])], title="t", width=360))


def text_box(t: ET.Element) -> tuple[float, float, float, float]:
    """Approximate box of an unrotated <text> element (same metrics the renderer uses)."""
    return _text_box(
        float(t.get("x", "0")),
        float(t.get("y", "0")),
        "".join(t.itertext()),
        t.get("text-anchor", "start"),
        float(t.get("font-size", "11")),
        t.get("font-weight") == "600",
    )


def text_el(root: ET.Element, label: str) -> ET.Element:
    return next(t for t in root.iter(f"{NS}text") if "".join(t.itertext()) == label)


def path_segments(d: str) -> list[tuple[float, float, float, float]]:
    """Segments of an M/L/H/V path."""
    import re

    segs, cur = [], (0.0, 0.0)
    for cmd, args in re.findall(r"([MLHV])([^MLHV]*)", d):
        nums = [float(v) for v in args.split()]
        if cmd == "M" or cmd == "L":
            nxt = (nums[0], nums[1])
        elif cmd == "H":
            nxt = (nums[0], cur[1])
        else:
            nxt = (cur[0], nums[0])
        if cmd != "M":
            segs.append((*cur, *nxt))
        cur = nxt
    return segs


def test_marker_label_nudge() -> None:
    svg = xy_chart(
        [],
        title="crowded",
        markers=[Marker(1, 1.0 + i * 1e-4, f"label {i}") for i in range(5)],
        x_range=(0, 2),
        y_range=(0, 2),
    )
    root = parse(svg)
    boxes = [text_box(t) for t in root.iter(f"{NS}text") if "label" in (t.text or "")]
    assert len(boxes) == 5
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            assert _overlap(a, b) == 0, (a, b)


def test_diagonal_label_off_the_line() -> None:
    bins = [(0.55, 0.5, 0.4, 0.6), (0.85, 0.83, 0.77, 0.88), (0.95, 0.97, 0.93, 0.99)]
    for w, h in ((640, 360), (420, 300), (360, 360)):
        svg = xy_chart(
            [Series("observed", [(b[0], b[1]) for b in bins], show_points=True)],
            title="Reliability",
            x_range=(0, 1),
            y_range=(0, 1),
            y_format="percent",
            error_bars=[ErrorBar(*b) for b in bins],
            diagonal=Diagonal("perfect calibration"),
            legend="none",
            width=w,
            height=h,
        )
        root = parse(svg)
        line = next(e for e in root.iter(f"{NS}line") if e.get("stroke-opacity") == "0.45")
        x0, y0, x1, y1 = (float(line.get(k, "0")) for k in ("x1", "y1", "x2", "y2"))
        t = text_el(root, "perfect calibration")
        tx, ty = float(t.get("x", "0")), float(t.get("y", "0"))
        assert t.get("text-anchor") == "end"
        ang = math.radians(float(t.get("transform", "").split("(")[1].split()[0]))
        ux, uy = math.cos(ang), math.sin(ang)
        assert abs(ux - (x1 - x0) / math.hypot(x1 - x0, y1 - y0)) < 1e-3  # along the line
        nx, ny = -uy, ux
        wdt = _text_w("perfect calibration", 10.5)
        corners = [
            (tx - a * ux + b * nx, ty - a * uy + b * ny) for a in (0, wdt) for b in (-8.6, 2.6)
        ]
        # signed distance of every glyph-box corner to the diagonal: same side, >= 3px away
        dists = [((cx - x0) * nx + (cy - y0) * ny) for cx, cy in corners]
        assert all(d >= 3 for d in dists) or all(d <= -3 for d in dists), dists
        # and inside the plot area
        assert all(x0 - 1 <= cx <= x1 + 1 and y1 - 1 <= cy <= y0 + 1 for cx, cy in corners)


def bar_boxes(svg: str, ref: str) -> tuple[ET.Element, tuple, list[tuple]]:
    root = parse(svg)
    ref_box = text_box(text_el(root, ref))
    anns = [text_box(t) for t in root.iter(f"{NS}text") if (t.text or "").startswith("n=")]
    return root, ref_box, anns


def test_bar_ref_label_avoids_annotations() -> None:
    svg = bar_with_ci(
        ["0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"],
        [0.12, 0.08, None, 0.03, 0.01],
        [0.05, 0.03, None, 0.01, 0.0],
        [0.25, 0.15, None, 0.07, 0.03],
        title="Disagreement by bin",
        y_format="percent",
        ref_line=RefLine("y", 0.05, "tolerance"),
        annotations=["n=20", "n=31", "n=0", "n=60", "n=200"],
    )
    _, ref_box, anns = bar_boxes(svg, "tolerance")
    assert len(anns) == 5
    assert all(_overlap(ref_box, a) == 0 for a in anns)
    # no free spot at all: every bar ends right at the tolerance line; annotations get nudged
    svg = bar_with_ci(
        ["a", "b", "c"],
        [0.05, 0.05, 0.05],
        title="crowded",
        y_format="percent",
        ref_line=RefLine("y", 0.05, "a rather long tolerance label"),
        annotations=["n=1000", "n=2000", "n=3000"],
        width=240,
        height=200,
    )
    _, ref_box, anns = bar_boxes(svg, "a rather long tolerance label")
    assert len(anns) == 3
    assert all(_overlap(ref_box, a) == 0 for a in anns)


def test_pareto_marker_labels_avoid_lines_and_each_other() -> None:
    names = ("only:fast", "only:slow", "oracle", "recommended")
    for kw in ({}, {"x_log": True, "width": 420, "height": 320}, {"x_log": True}):
        svg = pareto_svg(**kw)
        root = parse(svg)
        boxes = {n: text_box(text_el(root, n)) for n in names}
        frontier = next(p for p in root.iter(f"{NS}path") if "stroke-linejoin" in p.attrib).get(
            "d", ""
        )
        segs = path_segments(frontier)
        assert segs
        for n, b in boxes.items():
            assert sum(_clip_len(b, s) for s in segs) == 0, (kw, n)
        vals = list(boxes.values())
        for i, a in enumerate(vals):
            for b in vals[i + 1 :]:
                assert _overlap(a, b) == 0
        # no leader lines when there is room next to each marker (the old "stray dash")
        group = next(g for g in root.iter(f"{NS}g") if g.get("class") == "sg-markers")
        assert not list(group.iter(f"{NS}line")), kw
        for t in group.iter(f"{NS}text"):
            assert "".join(t.itertext()) in names  # no dash glyphs in labels


def test_risk_coverage_ref_label_avoids_step_line() -> None:
    pts = [(i / 20, 0.02 + (i / 20) ** 2 * 0.2) for i in range(1, 21)]
    svg = line_chart(
        Series("risk", pts, kind="step"),
        title="Risk-coverage",
        y_format="percent",
        ref_lines=[RefLine("y", 0.05, "tolerance 5%")],
        legend="none",
    )
    root = parse(svg)
    box = text_box(text_el(root, "tolerance 5%"))
    path = next(p for p in root.iter(f"{NS}path") if "stroke-linejoin" in p.attrib)
    assert sum(_clip_len(box, s) for s in path_segments(path.get("d", ""))) == 0


def test_determinism() -> None:
    assert pareto_svg() == pareto_svg()
    a = bar_with_ci(["x", "y"], [0.1, 0.2], [0.0, 0.1], [0.2, 0.3], title="d")
    b = bar_with_ci(["x", "y"], [0.1, 0.2], [0.0, 0.1], [0.2, 0.3], title="d")
    assert a == b


def test_clip_ids_differ_between_charts() -> None:
    a = line_chart(Series("s", [(0, 0), (1, 1)]), title="same")
    b = line_chart(Series("s", [(0, 0), (1, 2)]), title="same")
    ida = ET.fromstring(a).find(f"{NS}clipPath").get("id")  # type: ignore[union-attr]
    idb = ET.fromstring(b).find(f"{NS}clipPath").get("id")  # type: ignore[union-attr]
    assert ida != idb


def test_styles_and_errors() -> None:
    with pytest.raises(ValueError):
        line_chart(Series("s", [(0, 0)], style="nope"), title="t")
    with pytest.raises(ValueError):
        line_chart(Series("s", [(0, 0)]), title="t", width=10)
    with pytest.raises(ValueError):
        line_chart(Series("s", [(1, 1)]), title="t", x_range=(2, 1))
    svg = line_chart([Series(str(i), [(0, i), (1, i + 1)]) for i in range(10)], title="t")
    parse(svg)
    assert "--sg-series-8" in svg


def test_chart_css() -> None:
    assert ".sg-chart" in CHART_CSS
    assert "prefers-color-scheme: dark" in CHART_CSS
    for var in ("--sg-series-1", "--sg-series-8", "--sg-accent", "--sg-muted", "--sg-bg"):
        assert CHART_CSS.count(f"{var}:") == 3
