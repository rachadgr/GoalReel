#!/usr/bin/env bash
set -euo pipefail
python -m goalreel.cli.main analyze "${1:?video path}" --out "${2:-runs/source}"
python -m goalreel.cli.main render-vertical "${1}" --out "${2:-runs/source}/render/baseline.mp4"
