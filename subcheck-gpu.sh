#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
  echo "Virtual environment not found. Please create it first:"
  echo "  python3 -m venv .venv"
  echo "  source .venv/bin/activate"
  echo "  pip install -e . nvidia-cublas-cu12 'nvidia-cudnn-cu12==9.*'"
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# Auto-configure LD_LIBRARY_PATH for nvidia-* pip packages
CUDA_LIB_PATH="$(python3 -c '
import glob, site
dirs = site.getsitepackages()
lib_paths = []
for d in dirs:
    lib_paths.extend(glob.glob(d + "/nvidia/*/lib"))
print(":".join(sorted(set(lib_paths))))
' 2>/dev/null || true)"

if [ -n "$CUDA_LIB_PATH" ]; then
  export LD_LIBRARY_PATH="${CUDA_LIB_PATH}:${LD_LIBRARY_PATH:-}"
fi

# Run subcheck with CUDA device
exec python3 -m subcheck.cli --device cuda "$@"
