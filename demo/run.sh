#!/usr/bin/env bash
set -euo pipefail; cd "$(dirname "$0")/.."
if [ "${1:-}" = remake ]; then "${PYTHON:-python3}" example/make_example.py; fi
"${PYTHON:-python3}" demo/analyse.py && "${PYTHON:-python3}" demo/modes.py && "${PYTHON:-python3}" demo/electronic.py && "${PYTHON:-python3}" demo/statistics.py && "${PYTHON:-python3}" demo/plot.py && "${PYTHON:-python3}" demo/schematic.py
