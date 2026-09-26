#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO
CACHE="${COMPASS_DATA_CACHE:-$REPO/.oc_cache}/data"
MIRROR="https://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data"

mkdir -p "$CACHE"
if [[ -f "$CACHE/piqa/dev.jsonl" ]]; then
  echo "piqa   already present"
else
  echo "piqa   downloading"
  curl -fsSL "$MIRROR/piqa.zip" -o "$CACHE/piqa.zip"
  ( cd "$CACHE" && unzip -oq piqa.zip && rm -f piqa.zip )
fi

if [[ -f "$REPO/data/gpqa/gpqa_diamond.csv" ]]; then
  echo "gpqa   already present"
else
  echo "gpqa   downloading from the Hub (gated: needs an accepted licence)"
  mkdir -p "$REPO/data/gpqa"
  "${OC_PYTHON:-${PY:-python}}" - <<'PY'
import os, shutil
from huggingface_hub import hf_hub_download
dst = os.path.join(os.environ["REPO"], "data", "gpqa")
for name in ("gpqa_diamond.csv", "gpqa_main.csv"):
    shutil.copy(hf_hub_download("Idavidrein/gpqa", name, repo_type="dataset"),
                os.path.join(dst, name))
PY
fi

echo "piqa   $(ls "$CACHE/piqa" 2>/dev/null | tr '\n' ' ')"
echo "gpqa   $(ls "$REPO/data/gpqa" 2>/dev/null | tr '\n' ' ')"
