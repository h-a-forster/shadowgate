#!/usr/bin/env sh
# Reproduce the Haiku 5.5 -> Opus 5.5 experiment. Needs a logged-in Claude Code CLI.
# The eval run makes ~400 model calls. The serve run reuses them from the response cache.
set -eu
cd "$(dirname "$0")"
shadowgate datasets make arithmetic -n 200 --seed 2026 --min-steps 6 --max-steps 12 -o tasks.jsonl
shadowgate run -c shadowgate.toml -t tasks.jsonl --mode eval --run-id eval
shadowgate run -c shadowgate.toml -t tasks.jsonl --mode serve --run-id serve
shadowgate sweep --ledger ledger.sqlite --run-id eval
shadowgate export --ledger ledger.sqlite --run-id eval -o decisions-eval.jsonl
shadowgate export --ledger ledger.sqlite --run-id serve -o decisions-serve.jsonl
shadowgate report --ledger ledger.sqlite --run-id serve --sweep-run-id eval \
  -o ../../docs/results/arithmetic-haiku-opus.html
