#!/usr/bin/env bash
# Example data -> descriptors and voids -> figures (demo/out/).  bash demo/run.sh [remake]
set -euo pipefail; cd "$(dirname "$0")/.."
if [ "${1:-}" = remake ]; then python3 example/make_example.py; fi
python3 demo/analyse.py && python3 demo/modes.py && python3 demo/electronic.py && python3 demo/trends.py && python3 demo/plot.py && python3 demo/schematic.py
