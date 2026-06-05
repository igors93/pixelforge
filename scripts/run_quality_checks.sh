#!/usr/bin/env sh
set -eu

python -m pytest
python -m ruff check .
python -m mypy src
