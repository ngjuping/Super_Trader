#!/bin/sh
# Prints fail-closed preflight only. Does not export or echo credentials.
set -eu
cd "$(dirname "$0")/.."
exec python -m super_trader crypto-xs-live --config "${1:-config/crypto_xs_live.example.json}" --preflight
