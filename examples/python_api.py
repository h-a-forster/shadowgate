"""Build and audit a two-tier cascade from Python, fully offline.

    uv run python examples/python_api.py

Two simulated models stand in for a cheap fast model and an expensive strong one. The fast
tier states its confidence; answers below 0.8 escalate. A share of the accepted answers is
re-answered by the strong tier (shadow audit), and the summary estimates how often the fast
tier is wrong on the cases it kept.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import shadowgate as sg
from shadowgate.backends import SimulatedBackend
from shadowgate.compare import Numeric
from shadowgate.confidence import Verbal
from shadowgate.datasets import arithmetic
from shadowgate.extract import FinalLine
from shadowgate.pricing import Pricing

tasks = arithmetic(300, seed=0)  # generated word problems with known answers

# Simulated models know the tasks so they can answer "correctly" with a chosen skill.
# For real models, swap in e.g.:
#   from shadowgate.backends import AnthropicBackend   # pip install "shadowgate-llm[anthropic]"
#   fast_model = AnthropicBackend("claude-haiku-5-5")  # key read from ANTHROPIC_API_KEY
#   strong_model = AnthropicBackend("claude-opus-5-5")
fast_model = SimulatedBackend(
    "fast", skill=8.0, tasks=tasks, overconfidence=0.1, pricing=Pricing(0.5, 2.5)
)
strong_model = SimulatedBackend("strong", skill=16.0, tasks=tasks, pricing=Pricing(5.0, 25.0))

cascade = sg.Cascade(
    [
        sg.Tier("fast", fast_model, threshold=0.8, estimator=Verbal()),
        sg.Tier("strong", strong_model),  # final tier: always serves
    ],
    extractor=FinalLine(),  # the answer is the text after the last "ANSWER:"
    comparator=Numeric(),  # 1,000 == 1000.0
    # Audit borderline acceptances more often; weights 1/pi keep the estimate unbiased.
    audit=sg.AuditPolicy(rate=0.2, strata=((0.8, 0.9, 0.5), (0.9, 1.0, 0.2)), floor=0.05),
)

with tempfile.TemporaryDirectory() as tmp, sg.Ledger(Path(tmp) / "ledger.sqlite") as ledger:
    stats = sg.run_tasks(cascade, tasks, ledger, run_id="demo", workers=4)
    print(stats.summary_line())
    summary = sg.summarize(ledger.decisions("demo"), tolerance=0.05)

try:
    from shadowgate.report import render_text
except ImportError:
    render_text = None

if render_text is not None:
    print(render_text(summary))
else:
    d = summary.disagreement
    print(f"skipped cases: {summary.n_skipped}/{summary.n_decisions}, audited: {summary.n_audited}")
    if d is not None and d.value is not None:
        print(f"skipped-case disagreement: {d.value:.1%} (95% CI {d.lo:.1%} to {d.hi:.1%})")
    if summary.est_savings is not None and summary.est_savings.value is not None:
        print(f"estimated savings vs strong-only: {summary.est_savings.value:.1%}")
    print(f"status at tolerance 5%: {summary.status}")
