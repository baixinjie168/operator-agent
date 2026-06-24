#!/usr/bin/env bash
# start.sh — cross-platform launcher for operator-agent
# Works on Ubuntu/Linux (.venv/bin) and Windows Git Bash/MINGW (.venv/Scripts).
# Mirrors start.bat: activate venv, set PYTHONPATH, clean .pyc, run uvicorn

set -euo pipefail

# Switch to script directory (project root), resolving symlinks
cd "$(dirname "$(readlink -f "$0" 2>/dev/null || echo "$0")")"

# Activate virtual environment — detect layout (Linux: bin, Windows: Scripts)
if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source .venv/bin/activate
elif [[ -f ".venv/Scripts/activate" ]]; then
    # shellcheck source=/dev/null
    source .venv/Scripts/activate
else
    echo "[ERROR] .venv not found." >&2
    echo "         Run 'uv sync' or 'python3 -m venv .venv && pip install -r requirements.txt' first." >&2
    exit 1
fi

# Set PYTHONPATH (src directories of all three packages)
PYTHONPATH="$(pwd)/packages/shared/src:$(pwd)/packages/mcp-server/src:$(pwd)/packages/agent/src"
export PYTHONPATH

# Clean all .pyc compiled cache (skip .venv and .git).
# `|| true` so a non-zero find exit (common on Windows/MINGW due to
# permission/junction errors) does not abort the script under `set -e`.
find . \( -path './.venv' -o -path './.git' \) -prune -o -name '*.pyc' -type f -delete 2>/dev/null || true

# Start FastAPI server
echo "========================================================"
echo "  operator-agent  starting..."
echo "  PYTHONPATH = ${PYTHONPATH}"
echo "  URL        = http://127.0.0.1:8000"
echo "========================================================"
echo

exec python -m uvicorn agent.main:create_app --factory --host 127.0.0.1 --port 8000
