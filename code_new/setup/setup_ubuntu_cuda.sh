#!/usr/bin/env bash
# SurgicalSlicing - Ubuntu setup with an NVIDIA (CUDA) GPU.
#
#   bash code_new/setup/setup_ubuntu_cuda.sh               # venv in <repo>/.venv-cuda
#   bash code_new/setup/setup_ubuntu_cuda.sh --venv ~/venvs/surgical
#   bash code_new/setup/setup_ubuntu_cuda.sh --cpu         # same machine without a GPU
#
# Picks the PyTorch build from the CUDA version the installed driver supports
# (`nvidia-smi` -> "CUDA Version: X.Y"). Only the driver is needed: the CUDA runtime and
# cuDNN come inside the PyTorch wheels, no system CUDA toolkit is required.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$REPO/.venv-cuda"
CPU=0
PYTHON="${PYTHON:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv) VENV="$2"; shift 2 ;;
    --cpu)  CPU=1; shift ;;
    --python) PYTHON="$2"; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1"; exit 2 ;;
  esac
done

say() { echo -e "\033[1;36m[setup]\033[0m $*"; }
die() { echo -e "\033[1;31m[setup]\033[0m $*" >&2; exit 1; }

# ── Python ────────────────────────────────────────────────────────────────────────────
command -v "$PYTHON" >/dev/null || die "$PYTHON not found: sudo apt install python3 python3-venv python3-pip"
PYV=$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
  || die "Python $PYV is too old, need 3.10+ (Ubuntu 22.04 ships 3.10, 24.04 ships 3.12)"
"$PYTHON" -c 'import venv, ensurepip' 2>/dev/null \
  || die "venv module missing: sudo apt install python3-venv  (or python${PYV}-venv)"
say "Python $PYV"

# ── GPU / driver ──────────────────────────────────────────────────────────────────────
INDEX=""
if [[ $CPU -eq 1 ]]; then
  INDEX="https://download.pytorch.org/whl/cpu"
  say "CPU build requested"
else
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found - install the NVIDIA driver first:
    sudo ubuntu-drivers install        (then reboot)
  or run with --cpu"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
    || die "nvidia-smi failed - is the driver loaded? (reboot after installing it)"
  CUDA_DRV=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9]*\.[0-9]*\).*/\1/p' | head -1)
  [[ -n "$CUDA_DRV" ]] || die "could not read 'CUDA Version' from nvidia-smi"
  MAJ=${CUDA_DRV%%.*}; MIN=${CUDA_DRV##*.}; V=$((MAJ * 100 + MIN))
  say "driver supports CUDA up to $CUDA_DRV"
  # The default PyPI wheel of current PyTorch is built for CUDA 13.x (needs driver >= 580).
  # Older drivers get the matching wheel index; pip then takes the newest torch built for it.
  if   (( V >= 1300 )); then INDEX="";                                        say "using the default PyPI build (CUDA 13)"
  elif (( V >= 1208 )); then INDEX="https://download.pytorch.org/whl/cu128"; say "using the cu128 build"
  elif (( V >= 1206 )); then INDEX="https://download.pytorch.org/whl/cu126"; say "using the cu126 build"
  elif (( V >= 1108 )); then INDEX="https://download.pytorch.org/whl/cu118"; say "using the cu118 build (older torch)"
  else die "driver too old (CUDA $CUDA_DRV). Update it: sudo ubuntu-drivers install"
  fi
fi

# ── venv + packages ───────────────────────────────────────────────────────────────────
if [[ ! -x "$VENV/bin/python" ]]; then
  say "creating venv $VENV"
  "$PYTHON" -m venv "$VENV"
fi
PY="$VENV/bin/python"
"$PY" -m pip install --upgrade pip
if [[ -n "$INDEX" ]]; then
  "$PY" -m pip install --upgrade torch torchvision --index-url "$INDEX"
else
  "$PY" -m pip install --upgrade torch torchvision
fi
"$PY" -m pip install -r "$REPO/code_new/requirements.txt"

"$PY" "$REPO/code_new/setup/check_env.py" $([[ $CPU -eq 1 ]] || echo --require-cuda)

say "done. Activate and train:"
echo "    source \"$VENV/bin/activate\""
echo "    python code_new/train_surgical.py --config code_new/configs/definitor_gpu.yaml"
echo "    python code_new/train_surgical.py --config code_new/configs/classifier_gpu.yaml"
