#!/usr/bin/env bash
set -euo pipefail
export SMOKE_ROOT="${SMOKE_ROOT:-/scratch1/$(id -un)/qwen38-27b-llamafactory-test}"
export UV_CACHE_DIR="$SMOKE_ROOT/cache/uv" UV_PYTHON_INSTALL_DIR="$SMOKE_ROOT/python"
export PIP_CACHE_DIR="$SMOKE_ROOT/cache/pip" HF_HOME="$SMOKE_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" HF_DATASETS_CACHE="$HF_HOME/datasets"
export XDG_CACHE_HOME="$SMOKE_ROOT/cache" TORCH_HOME="$SMOKE_ROOT/cache/torch"
export TORCH_EXTENSIONS_DIR="$SMOKE_ROOT/cache/torch_extensions" TRITON_CACHE_DIR="$SMOKE_ROOT/cache/triton"
export TMPDIR="$SMOKE_ROOT/tmp" PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX="$SMOKE_ROOT/cache/pycache"
export UV_NO_CONFIG=1
export UV_PYTHON_BIN_DIR="$SMOKE_ROOT/bin"
mkdir -p "$TMPDIR" "$SMOKE_ROOT/bin" "$SMOKE_ROOT/logs" "$SMOKE_ROOT/data" "$UV_CACHE_DIR"
if [[ "${1:-}" == --run ]]; then
    shift
    exec "$SMOKE_ROOT/venv/bin/python" "$(dirname "$(realpath "$0")")/smoke_test.py" "$@"
fi
exec > >(tee -a "$SMOKE_ROOT/logs/setup.log") 2>&1
if [[ ! -x "$SMOKE_ROOT/bin/uv" ]]; then
    curl --proto '=https' --tlsv1.2 -LsSf https://astral.sh/uv/0.9.11/install.sh -o "$TMPDIR/install-uv.sh"
    UV_UNMANAGED_INSTALL="$SMOKE_ROOT/bin" sh "$TMPDIR/install-uv.sh"
fi
UV="$SMOKE_ROOT/bin/uv"
"$UV" python install 3.12.12
if [[ ! -x "$SMOKE_ROOT/venv/bin/python" ]]; then
    "$UV" venv --python 3.12.12 "$SMOKE_ROOT/venv"
fi
LF_COMMIT=8ebba0d07dd14fdb1d429c497f018c7dae2e44b7
LF="$SMOKE_ROOT/LLaMA-Factory"
if [[ ! -d "$LF/.git" ]]; then git clone https://github.com/hiyouga/LLaMA-Factory.git "$LF"; fi
if ! git -C "$LF" cat-file -e "$LF_COMMIT^{commit}"; then git -C "$LF" fetch origin "$LF_COMMIT"; fi
git -C "$LF" checkout --detach "$LF_COMMIT"
printf '%s\n' "$LF_COMMIT" > "$SMOKE_ROOT/llamafactory-commit.txt"
cat > "$SMOKE_ROOT/requirements.in" <<REQ
torch==2.10.0
torchvision==0.25.0
torchaudio==2.10.0
transformers==5.17.0
$LF
REQ
if [[ ! -f "$SMOKE_ROOT/requirements.lock" ]]; then
    REPO_LOCK="$(dirname "$(realpath "$0")")/requirements.lock"
    if [[ -f "$REPO_LOCK" ]]; then
        cp "$REPO_LOCK" "$SMOKE_ROOT/requirements.lock"
    else
        "$UV" pip compile --quiet --python "$SMOKE_ROOT/venv/bin/python" \
            --index https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match \
            "$SMOKE_ROOT/requirements.in" -o "$SMOKE_ROOT/requirements.lock"
    fi
fi
"$UV" pip sync --python "$SMOKE_ROOT/venv/bin/python" \
    --index https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match \
    "$SMOKE_ROOT/requirements.lock"
"$UV" pip check --python "$SMOKE_ROOT/venv/bin/python"
"$UV" pip freeze --python "$SMOKE_ROOT/venv/bin/python" > "$SMOKE_ROOT/installed.txt"
"$SMOKE_ROOT/venv/bin/python" "$(dirname "$(realpath "$0")")/smoke_test.py" preflight
