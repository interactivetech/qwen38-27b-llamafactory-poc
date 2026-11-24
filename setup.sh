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
REPO_DIR="$(dirname "$(realpath "$0")")"
if [[ "${1:-}" == --tracking ]]; then
    shift
    if [[ ! -x "$SMOKE_ROOT/tracking-venv/bin/python" ]]; then
        echo "Tracking environment missing. Run ./setup.sh --setup-tracking first." >&2
        exit 2
    fi
    exec "$SMOKE_ROOT/tracking-venv/bin/python" "$REPO_DIR/sft_tracking.py" "$@"
fi
TRACKING_ONLY=0
if [[ "${1:-}" == --setup-tracking ]]; then TRACKING_ONLY=1; fi
if [[ "${1:-}" == --run ]]; then
    shift
    if [[ "${1:-}" == terminalbench ]]; then
        shift
        if [[ ! -x "$SMOKE_ROOT/eval-harbor-venv/bin/python" ]]; then
            echo "Evaluation environment missing. Run ./setup.sh --setup-eval first." >&2
            exit 2
        fi
        exec "$SMOKE_ROOT/eval-harbor-venv/bin/python" "$REPO_DIR/terminalbench.py" "$@"
    fi
    exec "$SMOKE_ROOT/venv/bin/python" "$(dirname "$(realpath "$0")")/smoke_test.py" "$@"
fi
exec > >(tee -a "$SMOKE_ROOT/logs/setup.log") 2>&1
if [[ ! -x "$SMOKE_ROOT/bin/uv" ]]; then
    curl --proto '=https' --tlsv1.2 -LsSf https://astral.sh/uv/0.9.11/install.sh -o "$TMPDIR/install-uv.sh"
    UV_UNMANAGED_INSTALL="$SMOKE_ROOT/bin" sh "$TMPDIR/install-uv.sh"
fi
UV="$SMOKE_ROOT/bin/uv"
"$UV" python install 3.12.12
if [[ "${1:-}" == --setup-eval ]]; then
    for component in harbor vllm; do
        eval_env="$SMOKE_ROOT/eval-$component-venv"
        if [[ ! -x "$eval_env/bin/python" ]]; then
            "$UV" venv --python 3.12.12 "$eval_env"
        fi
        "$UV" pip sync --python "$eval_env/bin/python" "$REPO_DIR/requirements-eval-$component.lock"
        "$UV" pip check --python "$eval_env/bin/python"
        "$eval_env/bin/python" - "$REPO_DIR/requirements-eval-$component.lock" "$eval_env/ready.json" "$component" <<'PYEVAL'
import hashlib, importlib.metadata as m, json, sys
from pathlib import Path
component = sys.argv[3]
if component == "harbor":
    url = json.loads(m.distribution("harbor").read_text("direct_url.json"))
    assert url["vcs_info"]["commit_id"] == "39c2219a41b3caa7a689cef1882263b5a93ae93f"
    from harbor.environments.singularity.singularity import SingularityEnvironment
else:
    assert m.version("vllm") == "0.34.0"
    assert m.version("flashinfer-python") == m.version("flashinfer-cubin") == "0.7.3"
    assert m.version("flashinfer-jit-cache") == "0.7.3+cu130"
    import vllm
Path(sys.argv[2]).write_text(json.dumps(dict(lock_sha256=hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest(),
    version=m.version(component))) + "\n")
PYEVAL
    done
    mkdir -p "$SMOKE_ROOT/eval-bin"
    if ! command -v singularity >/dev/null; then
        ln -sfn "$(command -v apptainer)" "$SMOKE_ROOT/eval-bin/singularity"
    fi
    exit 0
fi
setup_tracking() {
    if [[ ! -x "$SMOKE_ROOT/tracking-venv/bin/python" ]]; then
        "$UV" venv --python 3.12.12 "$SMOKE_ROOT/tracking-venv"
    fi
    "$UV" pip sync --python "$SMOKE_ROOT/tracking-venv/bin/python" "$REPO_DIR/requirements-tracking.lock"
    "$UV" pip check --python "$SMOKE_ROOT/tracking-venv/bin/python"
    "$SMOKE_ROOT/tracking-venv/bin/python" - "$REPO_DIR/requirements-tracking.lock" "$SMOKE_ROOT/tracking-venv/ready.json" <<'PYTRACK'
import hashlib, json, sys
from pathlib import Path
import wandb
# Check the settings our reporter requires, without opening any network connection.
wandb.Settings(init_timeout=10, finish_timeout=3, console="off", disable_code=True,
               disable_git=True, disable_job_creation=True, x_disable_stats=True, x_disable_meta=True)
Path(sys.argv[2]).write_text(json.dumps(dict(lock_sha256=hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest(),
                                          wandb_version=wandb.__version__)) + "\n")
PYTRACK
}
if [[ "$TRACKING_ONLY" == 1 ]]; then
    setup_tracking
    exit 0
fi
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

setup_tracking
