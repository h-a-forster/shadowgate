#!/usr/bin/env sh
# Reproduce the BIG-Bench Hard cascade experiment: Haiku (low effort) -> Opus and Haiku -> Sonnet.
# Needs a logged-in Claude Code CLI, shadowgate installed, and a clone of
# github.com/suzgunmirac/BIG-Bench-Hard (see make_tasks.py). Every CLI call goes through
# claude-budget (../mmlu-pro-cascade/claude_budget.py), which logs the CLI's reported cost to
# ../mmlu-pro-cascade/costs.jsonl and refuses calls past SHADOWGATE_BUDGET_USD (default 88).
# Eval runs make ~1560 calls (~$11 at list prices); serve runs are response-cache hits.
set -eu
cd "$(dirname "$0")"
BBH=${BBH:-/tmp/BIG-Bench-Hard}
mkdir -p .bin && ln -sf ../../mmlu-pro-cascade/claude_budget.py .bin/claude-budget
export PATH="$PWD/.bin:$PATH"
python3 make_tasks.py "$BBH" -n 20 --seed 2026 -o tasks.jsonl
for pair in opus sonnet; do
  shadowgate run -c haiku-$pair.toml -t tasks.jsonl --mode eval --run-id $pair-eval
  shadowgate run -c haiku-$pair.toml -t tasks.jsonl --mode serve --run-id $pair-serve
  shadowgate export --ledger ledger.sqlite --run-id $pair-eval -o decisions-$pair-eval.jsonl
  shadowgate export --ledger ledger.sqlite --run-id $pair-serve -o decisions-$pair-serve.jsonl
  shadowgate report --ledger ledger.sqlite --run-id $pair-serve --sweep-run-id $pair-eval \
    -o ../../docs/results/bbh-haiku-$pair.html
done
python3 ../analyze_cascade.py . --json analysis.json | tee analysis.txt
