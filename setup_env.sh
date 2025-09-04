#!/usr/bin/env bash
# setup_ml_env.sh — create/activate a venv and install core ML packages
# Usage:
#   ./setup_ml_env.sh                    # uses .venv, python3, PyTorch CPU wheels
#   ./setup_ml_env.sh --venv .mlenv      # custom venv path
#   ./setup_ml_env.sh --python python3.11
#   ./setup_ml_env.sh --cuda cu121       # install PyTorch for CUDA 12.1
# Notes:
#   Valid --cuda values include: cpu (default), cu118, cu121, cu122, etc.

set -euo pipefail

VENV=".venv"
PYTHON="python3"
CUDA="cpu"   # or cu118/cu121/cu122...

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)   VENV="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --cuda)   CUDA="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,40p' "$0"; exit 0 ;;
    *)
      echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# 1) Create venv if missing
if [[ ! -d "$VENV" ]]; then
  echo "[+] Creating venv at: $VENV using $PYTHON"
  "$PYTHON" -m venv "$VENV"
fi

# 2) Activate venv
# shellcheck disable=SC1090
source "$VENV/bin/activate"

# 3) Upgrade packaging tools
python -m pip install --upgrade pip setuptools wheel

# 4) Core scientific + data stack
BASE_PKGS=(
  numpy
  scipy
  pandas
  scikit-learn
  matplotlib
  seaborn
  tqdm
  requests
  pyyaml
  rich
  ipdb
)

echo "[+] Installing base packages"
pip install "${BASE_PKGS[@]}"

# 5) PyTorch (CPU by default, or pick a CUDA build)
echo "[+] Installing PyTorch (${CUDA})"
if [[ "$CUDA" == "cpu" ]]; then
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
  pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${CUDA}"
fi

# 6) Register a Jupyter kernel for this venv (optional, handy)
# KERNEL_NAME="$(basename "$VENV")"
# python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "Python ($KERNEL_NAME)"

echo "[✓] Done. Activate with:  source \"$VENV/bin/activate\""
python - <<'PY'
import sys, pkgutil
needed = ["numpy","scipy","pandas","sklearn","matplotlib","seaborn","tqdm","requests","yaml","rich","torch","torchvision"]
missing = [m for m in needed if not pkgutil.find_loader(m)]
print("Sanity check:", "OK" if not missing else f"Missing: {', '.join(missing)}", file=sys.stderr)
PY
