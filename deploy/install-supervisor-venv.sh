#!/bin/sh
# Create the interpreter named by deploy/crypto-xs-supervisor.service.
# Uses python3. Does not read credentials or contact Bybit.
set -eu
ROOT="${1:-/opt/super-trader}"
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required; /usr/bin/python is not used" >&2
  exit 1
fi
if ! python3 -c 'import venv' >/dev/null 2>&1; then
  echo "python3-venv is required before creating $ROOT/.venv" >&2
  exit 1
fi
python3 -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/python" -m pip install --upgrade pip
"$ROOT/.venv/bin/python" -m pip install -r "$ROOT/requirements.txt"
test -x "$ROOT/.venv/bin/python"
"$ROOT/.venv/bin/python" -c 'import sys; assert sys.version_info >= (3, 10)'
echo "interpreter=$ROOT/.venv/bin/python"
