#!/usr/bin/env sh
# Reproduce the MMLU-Pro cascade experiment: Haiku (low effort) -> Opus and Haiku -> Sonnet.
# Needs a logged-in Claude Code CLI, shadowgate installed, and a clone of
# github.com/TIGER-AI-Lab/MMLU-Pro (see make_tasks.py). Every CLI call goes through
# claude-budget (claude_budget.py), which logs the CLI's reported cost to costs.jsonl and
# refuses calls past SHADOWGATE_BUDGET_USD (default 88).
# Eval runs make ~5000 calls (~$47 CLI-reported, pilot included); serve runs are response-cache
# hits. The pilot has eval runs for both pairs and a serve run for Haiku -> Opus only, as in the
# committed decisions-pilot-*.jsonl.
set -eu
cd "$(dirname "$0")"
MMLU_PRO=${MMLU_PRO:-/tmp/MMLU-Pro}
mkdir -p .bin && ln -sf ../claude_budget.py .bin/claude-budget
export PATH="$PWD/.bin:$PATH"
python3 make_tasks.py "$MMLU_PRO" -n 20 --seed 1 -o pilot-tasks.jsonl
python3 make_tasks.py "$MMLU_PRO" -n 1680 --seed 2026 --skip pilot-tasks.jsonl -o tasks.jsonl
for pair in opus sonnet; do
  shadowgate run -c haiku-$pair.toml -t pilot-tasks.jsonl --mode eval --run-id pilot-$pair-eval
  shadowgate export --ledger ledger.sqlite --run-id pilot-$pair-eval \
    -o decisions-pilot-$pair-eval.jsonl
done
shadowgate run -c haiku-opus.toml -t pilot-tasks.jsonl --mode serve --run-id pilot-opus-serve
shadowgate export --ledger ledger.sqlite --run-id pilot-opus-serve -o decisions-pilot-opus-serve.jsonl
for pair in opus sonnet; do
  shadowgate run -c haiku-$pair.toml -t tasks.jsonl --mode eval --run-id $pair-eval
  shadowgate run -c haiku-$pair.toml -t tasks.jsonl --mode serve --run-id $pair-serve
  shadowgate export --ledger ledger.sqlite --run-id $pair-eval -o decisions-$pair-eval.jsonl
  shadowgate export --ledger ledger.sqlite --run-id $pair-serve -o decisions-$pair-serve.jsonl
  shadowgate report --ledger ledger.sqlite --run-id $pair-serve --sweep-run-id $pair-eval \
    -o ../../docs/results/mmlu-pro-haiku-$pair.html
done
python3 ../analyze_cascade.py . --json analysis.json | tee analysis.txt
