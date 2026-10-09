"""Task file loading and built-in task generators.

Task files
----------
``load_tasks`` reads ``.jsonl``, ``.json`` and ``.csv`` files. Each record needs a prompt
(field ``prompt``, or the aliases ``question`` / ``input``) and may carry an ``id`` and a
``reference`` (aliases ``answer`` / ``target`` / ``label``). When several alias fields are present
the first one in that order wins and the others are kept in ``meta``. Every other field goes to
``Task.meta``; a ``meta`` field holding an object is merged into ``meta`` (this is how files
written by ``save_tasks`` round-trip).

Reference coercion: strings are kept exactly as written. JSON integers become their decimal
string (``42`` -> ``"42"``), integral floats drop the fractional part (``42.0`` -> ``"42"``),
other floats use ``repr`` (``0.5`` -> ``"0.5"``), booleans become ``"true"`` / ``"false"``,
lists and objects are JSON-encoded, and ``null`` means "no reference". In CSV files every cell is
a string, so an empty ``reference`` cell means "no reference" and an empty ``id`` cell means
"generate one".

Missing ids are generated from the 1-based record position: ``t0001``, ``t0002`` ... (zero-padded
to at least four digits). Duplicate ids raise ``DatasetError``.

Generators
----------
``arithmetic`` builds multi-step word problems with a verifiable integer answer whose difficulty
grows with the number of steps; harder problems also contain distractor numbers. ``gsm8k`` reads
a local GSM8K-format JSONL file (nothing is downloaded).
"""

from __future__ import annotations

import csv
import io
import json
import math
import random
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from .errors import DatasetError
from .types import Task

__all__ = [
    "load_tasks",
    "save_tasks",
    "arithmetic",
    "apply_ops",
    "gsm8k",
    "GENERATORS",
    "make",
]

PROMPT_FIELDS = ("prompt", "question", "input")
REFERENCE_FIELDS = ("reference", "answer", "target", "label")

# --------------------------------------------------------------------------- loading


def _coerce_reference(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return str(int(value))
        return repr(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _record_to_task(record: Any, where: str, *, from_csv: bool = False) -> tuple[Task, bool]:
    """Convert one raw record; returns (task, id_was_given)."""
    if not isinstance(record, Mapping):
        raise DatasetError(f"{where}: expected an object, got {type(record).__name__}")
    rec = dict(record)
    if from_csv:
        # Empty cells carry no information in CSV; treat them as absent for id/reference.
        for key in ("id", *REFERENCE_FIELDS):
            if rec.get(key) == "":
                rec.pop(key)

    prompt_key = next((k for k in PROMPT_FIELDS if k in rec), None)
    if prompt_key is None:
        raise DatasetError(
            f"{where}: missing prompt (expected one of: {', '.join(PROMPT_FIELDS)})"
        )
    prompt = rec.pop(prompt_key)
    if not isinstance(prompt, str):
        raise DatasetError(f"{where}: field {prompt_key!r} must be a string")
    if not prompt.strip():
        raise DatasetError(f"{where}: field {prompt_key!r} is empty")

    ref_key = next((k for k in REFERENCE_FIELDS if k in rec), None)
    reference = _coerce_reference(rec.pop(ref_key)) if ref_key is not None else None

    raw_id = rec.pop("id", None)
    given = raw_id is not None
    if given:
        if isinstance(raw_id, bool) or not isinstance(raw_id, (str, int)):
            raise DatasetError(f"{where}: field 'id' must be a string or integer")
        task_id = str(raw_id)
        if not task_id.strip():
            raise DatasetError(f"{where}: field 'id' is empty")
    else:
        task_id = ""  # filled in by the caller once the record count is known

    meta: dict[str, Any] = {}
    inner = rec.pop("meta", None)
    if isinstance(inner, Mapping):
        meta.update(inner)
    elif inner is not None:
        meta["meta"] = inner
    meta.update(rec)
    return Task(id=task_id, prompt=prompt, reference=reference, meta=meta), given


def _iter_jsonl(text: str, path: Path) -> Iterator[tuple[Any, str]]:
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{path}:{lineno}"
        try:
            yield json.loads(line), where
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{where}: invalid JSON ({exc.msg})") from None


def _iter_json(text: str, path: Path) -> Iterator[tuple[Any, str]]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{path}:{exc.lineno}: invalid JSON ({exc.msg})") from None
    if isinstance(data, Mapping):
        if "tasks" not in data:
            raise DatasetError(f"{path}: expected a list of tasks or an object with 'tasks'")
        data = data["tasks"]
    if not isinstance(data, list):
        raise DatasetError(f"{path}: expected a list of tasks")
    for i, item in enumerate(data):
        yield item, f"{path}: tasks[{i}]"


def _iter_csv(text: str, path: Path) -> Iterator[tuple[Any, str]]:
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if not reader.fieldnames:
        return
    names = [n.strip() for n in reader.fieldnames]
    if len(set(names)) != len(names):
        raise DatasetError(f"{path}: duplicate column names in header")
    reader.fieldnames = names
    for row in reader:
        where = f"{path}:{reader.line_num}"
        if None in row:
            raise DatasetError(f"{where}: row has more cells than the header")
        if all((v or "").strip() == "" for v in row.values()):
            continue
        yield {k: ("" if v is None else v) for k, v in row.items()}, where


_READERS: dict[str, Callable[[str, Path], Iterator[tuple[Any, str]]]] = {
    ".jsonl": _iter_jsonl,
    ".json": _iter_json,
    ".csv": _iter_csv,
}


def load_tasks(path: str | Path, *, limit: int | None = None) -> list[Task]:
    """Load tasks from a ``.jsonl``, ``.json`` or ``.csv`` file (see module docstring)."""
    p = Path(path)
    if limit is not None and limit < 0:
        raise DatasetError(f"limit must be >= 0, got {limit}")
    reader = _READERS.get(p.suffix.lower())
    if reader is None:
        raise DatasetError(
            f"{p}: unsupported task file extension {p.suffix!r} (use .jsonl, .json or .csv)"
        )
    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise DatasetError(f"{p}: cannot read task file ({exc.strerror or exc})") from None
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DatasetError(f"{p}: not valid UTF-8 ({exc.reason})") from None

    parsed: list[tuple[Task, bool, str]] = []
    for record, where in reader(text, p):
        if limit is not None and len(parsed) >= limit:
            break
        task, given = _record_to_task(record, where, from_csv=reader is _iter_csv)
        parsed.append((task, given, where))

    width = max(4, len(str(len(parsed))))
    tasks: list[Task] = []
    seen: dict[str, str] = {}
    for pos, (task, given, where) in enumerate(parsed, start=1):
        if not given:
            task = Task(
                id=f"t{pos:0{width}d}", prompt=task.prompt, reference=task.reference,
                meta=task.meta,
            )
        if task.id in seen:
            raise DatasetError(
                f"{where}: duplicate task id {task.id!r} (first seen at {seen[task.id]})"
            )
        seen[task.id] = where
        tasks.append(task)
    return tasks


def save_tasks(tasks: Iterable[Task], path: str | Path) -> None:
    """Write tasks as UTF-8 JSONL (one ``Task.to_dict()`` per line), creating parent dirs."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        for task in tasks:
            fh.write(json.dumps(task.to_dict(), ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- arithmetic

MAX_VALUE = 5000  # every intermediate quantity stays in [1, MAX_VALUE]
MAX_STEPS = 12

_NAMES = (
    "Amara", "Ben", "Chen", "Diego", "Elif", "Farah", "Gabriel", "Hana", "Ivan", "Jamal",
    "Keiko", "Liam", "Mei", "Nadia", "Omar", "Priya", "Quinn", "Rosa", "Sven", "Tariq",
    "Uma", "Victor", "Wanjiru", "Xavier", "Yara", "Zoltan", "Aisha", "Bruno", "Chidi", "Dana",
    "Emeka", "Fatima", "Goran", "Ingrid", "Jun", "Kofi", "Lucia", "Mateo", "Noor", "Olga",
    "Pablo", "Rania", "Sanjay", "Tomas", "Ayesha", "Leilani", "Hiroshi", "Marta",
)

# Each theme: opening (with {name} and {n}), the item noun (plural), add/sub phrasings, the
# final question, and theme-specific distractor sentences (with {d}). Pronouns are avoided so
# the text never depends on anyone's gender.
_THEMES: tuple[dict[str, Any], ...] = (
    {
        "item": "loaves of bread",
        "short": "loaves",
        "open": "{name} runs a bakery and starts the morning with {n} loaves of bread on the "
        "shelves.",
        "add": (
            "A fresh batch of {k} loaves comes out of the oven and goes onto the shelves.",
            "A supplier delivers {k} more loaves, which {name} puts on the shelves.",
        ),
        "sub": (
            "Customers buy {k} loaves.",
            "{name} donates {k} loaves to a local shelter.",
        ),
        "question": "How many loaves of bread are on the shelves now?",
        "distractors": (
            "The bakery has been open for {d} years.",
            "Each loaf sells for {d} dollars.",
            "{name} arrived at the bakery {d} minutes before opening.",
        ),
    },
    {
        "item": "books",
        "short": "books",
        "open": "{name} works at a library and has a cart holding {n} books.",
        "add": (
            "Readers return {k} books, and {name} adds them to the cart.",
            "{name} takes {k} more books from the drop box and puts them on the cart.",
        ),
        "sub": (
            "{name} shelves {k} books from the cart.",
            "Patrons check out {k} books straight from the cart.",
        ),
        "question": "How many books are on the cart now?",
        "distractors": (
            "The library has {d} reading tables.",
            "Each shelf in the library holds about {d} books.",
            "{name} has worked at the library for {d} years.",
        ),
    },
    {
        "item": "eggs",
        "short": "eggs",
        "open": "{name} keeps chickens and has {n} eggs in the farm's cooler.",
        "add": (
            "{name} collects {k} more eggs from the henhouse and puts them in the cooler.",
            "A neighbour drops off {k} eggs, which go into the cooler.",
        ),
        "sub": (
            "{name} sells {k} eggs at the farm stand.",
            "{name} uses {k} eggs to bake cakes.",
        ),
        "question": "How many eggs are in the cooler now?",
        "distractors": (
            "The farm has {d} chickens.",
            "The farm stand is open {d} days a week.",
            "The cooler is kept at {d} degrees.",
        ),
    },
    {
        "item": "boxes",
        "short": "boxes",
        "open": "A warehouse managed by {name} has {n} boxes in stock.",
        "add": (
            "A truck delivers {k} more boxes to the warehouse.",
            "{name} receives a shipment of {k} boxes.",
        ),
        "sub": (
            "{name} ships {k} boxes to customers.",
            "{k} boxes are loaded onto an outgoing truck.",
        ),
        "question": "How many boxes are in stock now?",
        "distractors": (
            "The warehouse has {d} loading docks.",
            "Each box weighs {d} kilograms.",
            "{name}'s shift lasts {d} hours.",
        ),
    },
    {
        "item": "stamps",
        "short": "stamps",
        "open": "{name} collects stamps and has {n} stamps in an album.",
        "add": (
            "{name} buys {k} more stamps at a fair and adds them to the album.",
            "A friend gives {name} {k} stamps for the album.",
        ),
        "sub": (
            "{name} trades away {k} stamps from the album.",
            "{name} removes {k} damaged stamps from the album.",
        ),
        "question": "How many stamps are in the album now?",
        "distractors": (
            "The album has {d} pages.",
            "{name} started collecting {d} years ago.",
            "The oldest stamp in the album is {d} years old.",
        ),
    },
    {
        "item": "apples",
        "short": "apples",
        "open": "{name} works in an orchard and has picked {n} apples so far.",
        "add": (
            "{name} picks {k} more apples.",
            "A coworker hands over {k} apples to add to {name}'s pile.",
        ),
        "sub": (
            "{name} sets aside {k} bruised apples and throws them away.",
            "{name} sells {k} apples to a passing customer.",
        ),
        "question": "How many apples does {name} have now?",
        "distractors": (
            "The orchard has {d} rows of trees.",
            "Each basket can hold {d} apples.",
            "{name} takes a break after {d} minutes.",
        ),
    },
    {
        "item": "coins",
        "short": "coins",
        "open": "{name} saves coins in a jar and currently has {n} coins in it.",
        "add": (
            "{name} adds {k} coins to the jar.",
            "A relative gives {name} {k} coins, which go into the jar.",
        ),
        "sub": (
            "{name} takes {k} coins out of the jar to buy a snack.",
            "{name} spends {k} coins from the jar on a book.",
        ),
        "question": "How many coins are in the jar now?",
        "distractors": (
            "The jar is {d} centimetres tall.",
            "{name} has been saving for {d} weeks.",
            "{name}'s favourite number is {d}.",
        ),
    },
    {
        "item": "seedlings",
        "short": "seedlings",
        "open": "{name} runs a plant nursery and has {n} seedlings in the greenhouse.",
        "add": (
            "{name} sprouts {k} more seedlings in the greenhouse.",
            "A supplier delivers {k} seedlings to the greenhouse.",
        ),
        "sub": (
            "{name} sells {k} seedlings.",
            "{k} seedlings are moved out of the greenhouse and planted in a park.",
        ),
        "question": "How many seedlings are in the greenhouse now?",
        "distractors": (
            "The greenhouse is {d} metres long.",
            "{name} waters the plants every {d} hours.",
            "The nursery employs {d} gardeners.",
        ),
    },
)

_MUL_PHRASES = (
    "{name} then gets enough extra {item} to make the total exactly {k} times what it was.",
    "After that, the number of {item} becomes {k} times as large as it was.",
)
_DIV_PHRASES = (
    "{name} then divides all the {item} into {k} equal groups, keeps one group, and gives the "
    "other groups away.",
    "Next, the {item} are split evenly into {k} equal shares; {name} keeps exactly one share "
    "and gives away the rest.",
)
_FLOORDIV_PHRASES = (
    "{name} then makes {k} equal groups with as many {item} in each group as possible, keeps "
    "one group, and gives away everything else, including any leftovers.",
    "Next, {name} shares the {item} among {k} people so that everyone gets the same whole "
    "number and as many as possible; {name} keeps one person's share and gives away the rest, "
    "including any leftovers.",
)
_GENERIC_DISTRACTORS = (
    "It is a {d}-minute walk from {name}'s home to work.",
    "{name} is {d} years old.",
    "The temperature outside is {d} degrees.",
)


def apply_ops(start: int, ops: Iterable[Iterable[Any]]) -> int:
    """Evaluate a step list ``[[op, operand], ...]`` from ``start``.

    ``op`` is ``"add"``, ``"sub"``, ``"mul"``, ``"div"`` (exact division) or ``"floordiv"``
    (keep only full groups). Raises ``DatasetError`` on an unknown op or an inexact ``"div"``.
    """
    value = int(start)
    for op, operand in ops:
        k = int(operand)
        if op == "add":
            value += k
        elif op == "sub":
            value -= k
        elif op == "mul":
            value *= k
        elif op == "div":
            if value % k:
                raise DatasetError(f"inexact division {value} / {k}")
            value //= k
        elif op == "floordiv":
            value //= k
        else:
            raise DatasetError(f"unknown op {op!r}")
    return value


def _choose_op(rng: random.Random, value: int, step_index: int) -> tuple[str, int]:
    """Pick a feasible operation keeping the value in [1, MAX_VALUE] and integer."""
    candidates: list[tuple[str, int, float]] = []
    add_hi = min(99, MAX_VALUE - value)
    if add_hi >= 2:
        candidates.append(("add", rng.randint(2, add_hi), 3.0))
    if value >= 4:
        candidates.append(("sub", rng.randint(1, min(99, value - 2)), 3.0))
    muls = [k for k in (2, 3, 4, 5) if value * k <= MAX_VALUE]
    if muls and step_index > 0:
        candidates.append(("mul", rng.choice(muls), 1.5))
    divs = [k for k in (2, 3, 4, 5, 6, 8) if value % k == 0 and value // k >= 2]
    if divs and step_index > 0:
        candidates.append(("div", rng.choice(divs), 1.5))
    fdivs = [k for k in (3, 4, 5, 6, 7) if value % k != 0 and value // k >= 2]
    if fdivs and step_index > 0:
        candidates.append(("floordiv", rng.choice(fdivs), 0.75))
    ops, ks, weights = zip(*candidates, strict=True)
    i = rng.choices(range(len(ops)), weights=weights)[0]
    return ops[i], ks[i]


def _difficulty(steps: int, distractors: int, ops: list[list[Any]]) -> int:
    """Steps dominate; distractors and multiplicative/floor steps add a little."""
    hard_ops = sum(1 for op, _ in ops if op in ("mul", "div", "floordiv"))
    floor_ops = sum(1 for op, _ in ops if op == "floordiv")
    return 2 * steps + distractors + (1 if hard_ops >= 2 else 0) + floor_ops


def _level(steps: int) -> str:
    if steps <= 2:
        return "easy"
    if steps <= 4:
        return "medium"
    return "hard"


def _make_problem(rng: random.Random, steps: int) -> tuple[str, int, int, list[list[Any]], int]:
    theme = rng.choice(_THEMES)
    name = rng.choice(_NAMES)
    start = rng.randint(12, 150)

    ops: list[list[Any]] = []
    value = start
    for i in range(steps):
        op, k = _choose_op(rng, value, i)
        ops.append([op, k])
        value = apply_ops(value, [[op, k]])
        if not 1 <= value <= MAX_VALUE:  # pragma: no cover - guarded by _choose_op
            raise AssertionError(f"generator left bounds: {value}")

    sentences: list[str] = []
    for op, k in ops:
        if op == "add":
            tpl = rng.choice(theme["add"])
        elif op == "sub":
            tpl = rng.choice(theme["sub"])
        elif op == "mul":
            tpl = rng.choice(_MUL_PHRASES)
        elif op == "div":
            tpl = rng.choice(_DIV_PHRASES)
        else:
            tpl = rng.choice(_FLOORDIV_PHRASES)
        sentences.append(tpl.format(name=name, k=k, item=theme["short"]))

    # Distractors: none for short problems, increasingly likely as steps grow.
    n_distractors = 0
    if steps >= 3 and rng.random() < min(0.85, 0.2 * (steps - 2)):
        n_distractors = 2 if steps >= 6 and rng.random() < 0.5 else 1
    pool = list(theme["distractors"]) + list(_GENERIC_DISTRACTORS)
    for tpl in rng.sample(pool, n_distractors):
        d = rng.randint(2, 60)
        sentences.insert(rng.randint(0, len(sentences)), tpl.format(name=name, d=d))

    opening = theme["open"].format(name=name, n=start)
    question = theme["question"].format(name=name)
    prompt = " ".join(
        [
            opening,
            *sentences,
            question,
            "The events happen in the order described. Give the answer as a single integer.",
        ]
    )
    return prompt, start, value, ops, n_distractors


def arithmetic(
    n: int = 200, *, seed: int = 0, min_steps: int = 1, max_steps: int = 6
) -> list[Task]:
    """Generate ``n`` multi-step word problems with integer answers.

    Each problem starts from a quantity and applies ``steps`` operations (add, subtract,
    multiply, exact division, or keep-only-full-groups division) drawn uniformly from
    ``[min_steps, max_steps]``. Intermediate values stay integer and within ``[1, 5000]``.
    Problems with three or more steps may include distractor sentences with irrelevant numbers.

    ``meta`` holds ``source="arithmetic"``, ``seed``, ``steps``, ``difficulty`` (an integer that
    grows with ``steps``; distractors and multiplicative steps add a little), ``level``
    (easy/medium/hard), ``distractors``, ``start`` and ``ops`` (``[[op, operand], ...]``) so the
    reference can be re-derived with ``apply_ops(meta["start"], meta["ops"])``.
    Deterministic for a given argument set.
    """
    if n < 0:
        raise DatasetError(f"n must be >= 0, got {n}")
    if not 1 <= min_steps <= max_steps <= MAX_STEPS:
        raise DatasetError(
            f"need 1 <= min_steps <= max_steps <= {MAX_STEPS}, got {min_steps}..{max_steps}"
        )
    rng = random.Random(f"shadowgate-arithmetic:{seed}")
    width = max(4, len(str(n)))
    tasks: list[Task] = []
    for i in range(1, n + 1):
        steps = rng.randint(min_steps, max_steps)
        prompt, start, answer, ops, n_distractors = _make_problem(rng, steps)
        # Independent check: re-evaluate the step list from scratch.
        if apply_ops(start, ops) != answer:  # pragma: no cover - defensive
            raise AssertionError("arithmetic generator produced an inconsistent answer")
        tasks.append(
            Task(
                id=f"arith-{i:0{width}d}",
                prompt=prompt,
                reference=str(answer),
                meta={
                    "source": "arithmetic",
                    "seed": seed,
                    "steps": steps,
                    "difficulty": _difficulty(steps, n_distractors, ops),
                    "level": _level(steps),
                    "distractors": n_distractors,
                    "start": start,
                    "ops": ops,
                },
            )
        )
    return tasks


# --------------------------------------------------------------------------- gsm8k

def gsm8k(path: str | Path, *, limit: int | None = None) -> list[Task]:
    """Load a local GSM8K-format JSONL file (``{"question", "answer"}`` per line).

    The reference is the text after the final ``####`` with commas removed. The worked solution
    (text before ``####``) is kept in ``meta["solution"]``. Ids are ``gsm8k-0001``, ...
    """
    p = Path(path)
    if limit is not None and limit < 0:
        raise DatasetError(f"limit must be >= 0, got {limit}")
    try:
        text = p.read_bytes().decode("utf-8-sig")
    except OSError as exc:
        raise DatasetError(f"{p}: cannot read file ({exc.strerror or exc})") from None
    except UnicodeDecodeError as exc:
        raise DatasetError(f"{p}: not valid UTF-8 ({exc.reason})") from None

    rows: list[tuple[str, str, str]] = []
    for record, where in _iter_jsonl(text, p):
        if limit is not None and len(rows) >= limit:
            break
        if not isinstance(record, Mapping):
            raise DatasetError(f"{where}: expected an object")
        question, answer = record.get("question"), record.get("answer")
        if not isinstance(question, str) or not question.strip():
            raise DatasetError(f"{where}: missing 'question'")
        if not isinstance(answer, str):
            raise DatasetError(f"{where}: missing 'answer'")
        solution, sep, final = answer.rpartition("####")
        if not sep:
            raise DatasetError(f"{where}: answer has no '#### <number>' line")
        reference = final.strip().replace(",", "")
        if not reference:
            raise DatasetError(f"{where}: empty final answer after '####'")
        rows.append((question, reference, solution.strip()))

    width = max(4, len(str(len(rows))))
    return [
        Task(
            id=f"gsm8k-{i:0{width}d}",
            prompt=q,
            reference=ref,
            meta={"source": "gsm8k", "solution": sol},
        )
        for i, (q, ref, sol) in enumerate(rows, start=1)
    ]


# --------------------------------------------------------------------------- registry

GENERATORS: dict[str, Callable[..., list[Task]]] = {"arithmetic": arithmetic}


def make(name: str, **kwargs: Any) -> list[Task]:
    """Run the generator registered as ``name`` with ``kwargs``."""
    try:
        gen = GENERATORS[name]
    except KeyError:
        known = ", ".join(sorted(GENERATORS))
        raise DatasetError(f"unknown dataset generator {name!r} (known: {known})") from None
    try:
        return gen(**kwargs)
    except TypeError as exc:
        raise DatasetError(f"bad arguments for generator {name!r}: {exc}") from None
