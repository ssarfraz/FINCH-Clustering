#!/usr/bin/env bash
set -euo pipefail

# requires TWINE env vars set:
# export TWINE_USERNAME="__token__"
# export TWINE_PASSWORD="pypi-xxxxxxxxxxxxxxxx"

rm -rf dist build *.egg-info
python3 -m pip install --upgrade pip build twine
python3 -m build                  # makes sdist + wheel in dist/
python3 -m twine check dist/*     # sanity check
python3 -m twine upload dist/*    # upload both sdist and wheel
