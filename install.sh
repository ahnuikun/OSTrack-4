#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-track}"
ENV_FILE="${ENV_FILE:-ostrack_track_cu128_env.yaml}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found. Please install Anaconda/Miniconda and initialize conda first."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PATH="${SCRIPT_DIR}/${ENV_FILE}"

if [ ! -f "${ENV_PATH}" ]; then
  echo "Environment file not found: ${ENV_PATH}"
  exit 1
fi

echo "****************** OSTrack environment setup ******************"
echo "Environment name: ${ENV_NAME}"
echo "Environment file: ${ENV_PATH}"
echo ""

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "****************** Updating existing conda environment ******************"
  conda env update -n "${ENV_NAME}" -f "${ENV_PATH}" --prune
else
  echo "****************** Creating conda environment ******************"
  conda env create -n "${ENV_NAME}" -f "${ENV_PATH}"
fi

echo ""
echo "****************** Verifying PyTorch CUDA build ******************"
conda run -n "${ENV_NAME}" python - <<'PY'
import torch
import torchvision

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
PY

echo ""
echo "****************** Verifying tikzplotlib compatibility ******************"
conda run -n "${ENV_NAME}" python - <<'PY'
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tikzplotlib

plt.plot([1, 2], [3, 4])
out_file = Path(tempfile.gettempdir()) / "ostrack_tikzplotlib_check.tex"
tikzplotlib.save(str(out_file))
print("tikzplotlib export:", out_file.exists())
PY

echo ""
echo "****************** Installation complete! ******************"
echo "Activate it with: conda activate ${ENV_NAME}"
