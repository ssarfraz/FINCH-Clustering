#!/usr/bin/env bash
set -euo pipefail

# Build fresh
rm -rf dist build *.egg-info
python3 -m pip install --upgrade pip build twine
python3 -m build
python3 -m twine check dist/*

# Upload (use CA bundle if provided via env var)
if [[ -n "${REQUESTS_CA_BUNDLE:-}" ]]; then
  python3 -m twine upload --cert "$REQUESTS_CA_BUNDLE" dist/*
else
  python3 -m twine upload dist/*
fi
