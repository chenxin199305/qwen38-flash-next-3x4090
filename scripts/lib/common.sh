#!/bin/bash
# Shared helpers. Source from scripts; do not execute directly.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ENV_FILE:-$REPO_DIR/.env}"

if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

log()  { printf '\n==> %s\n' "$*"; }
warn() { printf '!! %s\n' "$*" >&2; }
die()  { printf '!! %s\n' "$*" >&2; exit 1; }

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

need_file() {
    [[ -e "$1" ]] || die "missing file: $1"
}

need_dir() {
    [[ -d "$1" ]] || die "missing directory: $1"
}

vllm_python() {
    local py="${VENV:-$HOME/vllm-venv}/bin/python"
    [[ -x "$py" ]] || die "vLLM Python not found: $py (run ./install.sh or set VENV)"
    printf '%s\n' "$py"
}

vllm_root() {
    local py
    py="$(vllm_python)"
    "$py" - <<'PY'
import pathlib
import vllm
print(pathlib.Path(vllm.__file__).parent)
PY
}

check_vllm_version() {
    local py expected actual
    py="$(vllm_python)"
    expected="${VLLM_VERSION:-0.30.0}"
    actual="$("$py" -c 'import vllm; print(vllm.__version__)')"
    [[ "$actual" == "$expected" ]] || die "vLLM $actual found, expected $expected. Patch anchors are version-specific."
}

gpu_count() {
    nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l
}

port_in_use() {
    local port="$1"
    if command -v ss >/dev/null 2>&1; then
        ss -ltn "sport = :$port" 2>/dev/null | grep -q LISTEN
    else
        return 1
    fi
}