from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from shadowgate.datasets import (
    GENERATORS,
    apply_ops,
    arithmetic,
    gsm8k,
    load_tasks,
    make,
    save_tasks,
)
from shadowgate.errors import DatasetError
from shadowgate.types import Task

# --------------------------------------------------------------------------- load / save


def _write(path: Path, text: str, encoding: str = "utf-8") -> Path:
    path.write_bytes(text.encode(encoding))
    return path


def test_save_load_round_trip(tmp_path: Path) -> None:
    tasks = [
        Task("a", "What is 2+2?", "4", {"difficulty": 1, "tags": ["x"]}),
        Task("b", "Ünïcödé prompt ✓", None, {}),
        Task("c", "line1\nline2", "", {"nested": {"k": 1}}),
    ]
    out = tmp_path / "deep" / "dir" / "tasks.jsonl"
    save_tasks(tasks, out)
    raw = out.read_bytes()
    assert b"\r\n" not in raw
    assert "Ünïcödé".encode() in raw  # ensure_ascii=False
    assert load_tasks(out) == tasks


def test_jsonl_aliases_meta_and_generated_ids(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "t.jsonl",
        '{"question": "Q1", "answer": 42, "subject": "math"}\r\n'
        "\r\n"
        '{"input": "Q2", "target": 42.0}\n'
        '{"prompt": "Q3", "label": 0.5}\n'
        '{"prompt": "Q4", "reference": "007", "question": "kept in meta"}\n'
        '{"prompt": "Q5", "reference": true, "id": 9}\n'
        '{"prompt": "Q6", "reference": null}\n',
    )
    tasks = load_tasks(p)
    assert [t.id for t in tasks] == ["t0001", "t0002", "t0003", "t0004", "9", "t0006"]
    assert [t.prompt for t in tasks] == ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
    assert [t.reference for t in tasks] == ["42", "42", "0.5", "007", "true", None]
    assert tasks[0].meta == {"subject": "math"}
    assert tasks[3].meta == {"question": "kept in meta"}


def test_jsonl_limit(tmp_path: Path) -> None:
    p = _write(tmp_path / "t.jsonl", "".join(f'{{"prompt": "q{i}"}}\n' for i in range(10)))
    assert [t.prompt for t in load_tasks(p, limit=3)] == ["q0", "q1", "q2"]
    assert load_tasks(p, limit=0) == []
    with pytest.raises(DatasetError):
        load_tasks(p, limit=-1)


def test_jsonl_bad_json_reports_line(tmp_path: Path) -> None:
    p = _write(tmp_path / "t.jsonl", '{"prompt": "ok"}\n\n{"prompt": oops}\n')
    with pytest.raises(DatasetError, match=r"t\.jsonl:3: invalid JSON"):
        load_tasks(p)


def test_missing_prompt_reports_location(tmp_path: Path) -> None:
    p = _write(tmp_path / "t.jsonl", '{"prompt": "ok"}\n{"answer": "4"}\n')
    with pytest.raises(DatasetError, match=r"t\.jsonl:2: missing prompt"):
        load_tasks(p)


@pytest.mark.parametrize("record", ['{"prompt": 5}', '{"prompt": "  "}', "[1, 2]"])
def test_invalid_records(tmp_path: Path, record: str) -> None:
    p = _write(tmp_path / "t.jsonl", record + "\n")
    with pytest.raises(DatasetError, match=r"t\.jsonl:1"):
        load_tasks(p)


def test_duplicate_ids(tmp_path: Path) -> None:
    p = _write(tmp_path / "t.jsonl", '{"id": "x", "prompt": "a"}\n{"id": "x", "prompt": "b"}\n')
    with pytest.raises(DatasetError, match=r"duplicate task id 'x'.*t\.jsonl:1"):
        load_tasks(p)


def test_generated_id_clash_is_duplicate(tmp_path: Path) -> None:
    p = _write(tmp_path / "t.jsonl", '{"prompt": "a"}\n{"id": "t0001", "prompt": "b"}\n')
    with pytest.raises(DatasetError, match="duplicate"):
        load_tasks(p)


def test_json_list_and_tasks_object(tmp_path: Path) -> None:
    records = [{"id": "a", "question": "Q", "answer": 3}, {"prompt": "R"}]
    p1 = _write(tmp_path / "list.json", json.dumps(records))
    p2 = _write(tmp_path / "obj.json", json.dumps({"tasks": records}, indent=2))
    for p in (p1, p2):
        tasks = load_tasks(p)
        assert [(t.id, t.prompt, t.reference) for t in tasks] == [
            ("a", "Q", "3"),
            ("t0002", "R", None),
        ]


def test_json_errors(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match=r"bad\.json:2: invalid JSON"):
        load_tasks(_write(tmp_path / "bad.json", '[\n{"prompt": }\n]'))
    with pytest.raises(DatasetError, match="tasks"):
        load_tasks(_write(tmp_path / "obj.json", '{"items": []}'))
    with pytest.raises(DatasetError, match=r"tasks\[1\]: missing prompt"):
        load_tasks(_write(tmp_path / "m.json", '[{"prompt": "a"}, {"answer": 1}]'))


def test_csv_with_bom_crlf_aliases(tmp_path: Path) -> None:
    text = (
        "id,question,answer,topic\r\n"
        "q1,\"What is 6, times 7?\",42,math\r\n"
        ",Name a colour,,art\r\n"
        "\r\n"
        "q3,\"multi\r\nline\",x,misc\r\n"
    )
    p = _write(tmp_path / "t.csv", text, encoding="utf-8-sig")
    tasks = load_tasks(p)
    assert [t.id for t in tasks] == ["q1", "t0002", "q3"]
    assert tasks[0].prompt == "What is 6, times 7?"
    assert tasks[0].reference == "42"
    assert tasks[0].meta == {"topic": "math"}
    assert tasks[1].reference is None
    assert tasks[2].prompt == "multi\r\nline"


def test_csv_errors(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match=r"t\.csv:3: missing prompt|t\.csv:3: field"):
        load_tasks(_write(tmp_path / "t.csv", "prompt,answer\nok,1\n,2\n"))
    with pytest.raises(DatasetError, match="missing prompt"):
        load_tasks(_write(tmp_path / "n.csv", "text,answer\nhello,1\n"))
    with pytest.raises(DatasetError, match="more cells"):
        load_tasks(_write(tmp_path / "x.csv", "prompt\na,b\n"))


def test_unknown_extension_and_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="extension"):
        load_tasks(_write(tmp_path / "t.txt", "hello"))
    with pytest.raises(DatasetError, match="cannot read"):
        load_tasks(tmp_path / "missing.jsonl")


# --------------------------------------------------------------------------- arithmetic


def _independent_eval(start: int, ops: list[list[object]]) -> int:
    """Re-derive an answer without using the library's evaluator."""
    v = start
    for op, k in ops:
        assert isinstance(k, int) and k > 0
        if op == "add":
            v = v + k
        elif op == "sub":
            v = v - k
        elif op == "mul":
            v = v * k
        elif op == "div":
            assert v % k == 0, "exact division step must not leave a remainder"
            v = v // k
        elif op == "floordiv":
            assert v % k != 0, "floor step should actually floor"
            v = (v - v % k) // k
        else:
            raise AssertionError(op)
        assert 1 <= v <= 5000
    return v


def test_arithmetic_answers_verified_independently() -> None:
    tasks = arithmetic(500, seed=123, min_steps=1, max_steps=8)
    assert len(tasks) == 500
    assert len({t.id for t in tasks}) == 500
    for t in tasks:
        m = t.meta
        assert m["steps"] == len(m["ops"])
        assert t.reference == str(_independent_eval(m["start"], m["ops"]))
        assert t.reference == str(apply_ops(m["start"], m["ops"]))
        # Every operand and the start value are stated in the prompt.
        numbers = set(re.findall(r"\d+", t.prompt))
        assert str(m["start"]) in numbers
        for _, k in m["ops"]:
            assert str(k) in numbers
        assert t.prompt.rstrip().endswith("single integer.")
        assert "?" in t.prompt


def test_arithmetic_deterministic_and_seed_sensitive() -> None:
    a = arithmetic(50, seed=7)
    assert a == arithmetic(50, seed=7)
    b = arithmetic(50, seed=8)
    assert [t.prompt for t in a] != [t.prompt for t in b]
    # A prefix is stable when n grows.
    assert arithmetic(60, seed=7)[:50] == a


def test_arithmetic_difficulty_grows_with_steps() -> None:
    tasks = arithmetic(600, seed=1, min_steps=1, max_steps=8)
    by_steps: dict[int, list[int]] = {}
    for t in tasks:
        by_steps.setdefault(t.meta["steps"], []).append(t.meta["difficulty"])
    means = [sum(v) / len(v) for _, v in sorted(by_steps.items())]
    assert sorted(by_steps) == list(range(1, 9))
    assert all(x < y for x, y in zip(means, means[1:], strict=False))
    # Steps dominate: any (s+1)-step problem is at least as hard as any s-step one minus slack.
    for s in range(1, 8):
        assert min(by_steps[s + 1]) > min(by_steps[s])
    # Distractors only on longer problems, and present in a good share of the hardest ones.
    assert all(t.meta["distractors"] == 0 for t in tasks if t.meta["steps"] <= 2)
    hard = [t for t in tasks if t.meta["steps"] >= 6]
    assert sum(t.meta["distractors"] > 0 for t in hard) > len(hard) / 2
    assert {t.meta["level"] for t in tasks} == {"easy", "medium", "hard"}


def test_arithmetic_step_range_and_validation() -> None:
    tasks = arithmetic(40, seed=3, min_steps=2, max_steps=2)
    assert {t.meta["steps"] for t in tasks} == {2}
    assert arithmetic(0) == []
    for kwargs in ({"min_steps": 0}, {"min_steps": 4, "max_steps": 3}, {"max_steps": 99}):
        with pytest.raises(DatasetError):
            arithmetic(5, **kwargs)
    with pytest.raises(DatasetError):
        arithmetic(-1)


def test_apply_ops_errors() -> None:
    assert apply_ops(10, [["add", 5], ["mul", 2], ["floordiv", 4], ["sub", 1]]) == 6
    with pytest.raises(DatasetError):
        apply_ops(10, [["div", 3]])
    with pytest.raises(DatasetError):
        apply_ops(10, [["pow", 2]])


def test_arithmetic_save_load_round_trip(tmp_path: Path) -> None:
    tasks = arithmetic(30, seed=5)
    save_tasks(tasks, tmp_path / "a.jsonl")
    assert load_tasks(tmp_path / "a.jsonl") == tasks


def test_committed_example_matches_generator() -> None:
    path = Path(__file__).resolve().parents[1] / "examples" / "tasks" / "arithmetic-50.jsonl"
    assert load_tasks(path) == arithmetic(50, seed=0)


def test_make_and_registry() -> None:
    assert "arithmetic" in GENERATORS
    assert make("arithmetic", n=5, seed=2) == arithmetic(5, seed=2)
    with pytest.raises(DatasetError, match="unknown dataset generator"):
        make("nope")
    with pytest.raises(DatasetError, match="bad arguments"):
        make("arithmetic", bogus=1)


# --------------------------------------------------------------------------- gsm8k


def test_gsm8k_parsing(tmp_path: Path) -> None:
    rows = [
        {"question": "Tom has 3 apples...", "answer": "He has 3+2=<<3+2=5>>5.\n#### 5"},
        {"question": "A big number?", "answer": "Work...\n#### 1,234"},
        {"question": "Negative?", "answer": "Steps\n####  -7 "},
    ]
    p = _write(
        tmp_path / "gsm.jsonl", "\r\n".join(json.dumps(r) for r in rows) + "\r\n\r\n"
    )
    tasks = gsm8k(p)
    assert [t.id for t in tasks] == ["gsm8k-0001", "gsm8k-0002", "gsm8k-0003"]
    assert [t.reference for t in tasks] == ["5", "1234", "-7"]
    assert tasks[0].prompt == "Tom has 3 apples..."
    assert tasks[0].meta["source"] == "gsm8k"
    assert tasks[0].meta["solution"] == "He has 3+2=<<3+2=5>>5."
    assert len(gsm8k(p, limit=2)) == 2


def test_gsm8k_errors(tmp_path: Path) -> None:
    p = _write(tmp_path / "g.jsonl", '{"question": "q", "answer": "no marker"}\n')
    with pytest.raises(DatasetError, match=r"g\.jsonl:1: .*####"):
        gsm8k(p)
    p = _write(tmp_path / "h.jsonl", '{"question": "q", "answer": "#### 1"}\n{bad\n')
    with pytest.raises(DatasetError, match=r"h\.jsonl:2: invalid JSON"):
        gsm8k(p)
