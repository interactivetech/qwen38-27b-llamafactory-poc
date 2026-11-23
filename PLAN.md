# Qwen3.8-27B LoRA smoke test on B300

## Summary

Install an isolated LLaMA-Factory environment under `/scratch1/mendeza`, then create a reproducible script that trains a LoRA adapter for **10 optimizer steps on 32 short dataset excerpts**, saves it, and verifies that it reloads.

Confirmed hardware: one B300 with approximately **269 GiB VRAM**, ample host RAM, and **26 TB free scratch space**. Start with BF16 LoRA; this should fit at short sequence lengths, subject to verification.

Use `Qwen/Qwen3.8-27B` and the confirmed dataset `shunzou05/TerminalHorizon-3K`. LLaMA-Factory currently supports the model through its `qwen3_8` template. See [supported models](https://github.com/hiyouga/LLaMA-Factory#supported-models).

## Environment and storage

- Use `/scratch1/mendeza/qwen38-27b-llamafactory-test/` for the Python environment, LLaMA-Factory checkout, caches, temporary files, prepared data, logs, and adapters. Keep only small scripts and configuration in the current workspace.
- Bootstrap `uv` and a managed Python **3.12** into scratch; create the virtual environment there. Avoid the existing home-directory Python installation.
- Set scratch locations for `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`, `PIP_CACHE_DIR`, `HF_HOME`, `HF_HUB_CACHE`, `HF_DATASETS_CACHE`, `XDG_CACHE_HOME`, `TORCH_HOME`, `TORCH_EXTENSIONS_DIR`, `TRITON_CACHE_DIR`, and `TMPDIR` before installation or imports. Disable user-site packages.
- Install the verified CUDA 13.0 wheel combination: **PyTorch 2.10.0, torchvision 0.25.0, torchaudio 2.10.0**.
- Check out and record a fixed LLaMA-Factory commit. Install it with **Transformers 5.17.0**, retaining its declared dependency constraints; save a complete resolved dependency lock. Its current dependency ceiling excludes the latest Transformers release. See the [dependency manifest](https://github.com/hiyouga/LLaMA-Factory/blob/main/pyproject.toml).
- Initially use PyTorch SDPA and the model's standard kernels. Avoid optional FlashAttention, DeepSpeed, quantization, and custom kernel builds for this short test.

## Scripts and training behavior

Create three small, reviewable files:

- **`setup.sh`**: idempotently prepare scratch directories, bootstrap the environment, install dependencies, and run dependency/CUDA checks.
- **`smoke_test.py`**: provide `prepare`, `train`, `verify`, and `all` commands; accept overrides for scratch root, example count, step count, and sequence length.
- **`smoke_lora.yaml`**: hold the LLaMA-Factory training defaults.

### Data preparation

- Stream the dataset rather than downloading its entire roughly 1.75 GB JSONL file.
- Select the first 32 eligible trajectories in stable source order. From each, retain the longest initial conversation prefix ending with a complete assistant turn that fits **2,048 tokens after template formatting**.
- Preserve message roles, reasoning text, JSON commands, and terminal observations. Register the local data using ShareGPT formatting with the dataset's `role` and `content` fields.
- Mask user messages and padding from loss; reject examples with no supervised tokens. Record source indices, token counts, and model/dataset revision hashes.

The sampled data contains alternating user/assistant conversations. No terminal commands from the dataset need to be executed. See the [dataset](https://huggingface.co/datasets/shunzou05/TerminalHorizon-3K).

### Training defaults

- Single GPU; BF16 model weights; SFT with `qwen3_8`.
- LoRA rank **8**, alpha **16**, dropout **0**; target supported language-model linear layers, keeping vision components and base weights frozen.
- Batch size **1**, gradient accumulation **1**, gradient checkpointing enabled, packing disabled, cutoff **2,048**, seed **42**.
- AdamW, learning rate **1e-4**, constant schedule, **10 optimizer steps**, loss logged every step.
- Save the final adapter and tokenizer; disable external experiment reporting and full-model merging.

## Validation and acceptance

- Before downloading weights: verify dependency consistency, CUDA availability, BF16 support, GPU architecture compatibility, and a CUDA matrix multiplication.
- Before model loading: check available GPU memory and fail clearly if another workload leaves insufficient capacity. Do not stop unrelated GPU processes.
- Before training: inspect formatted examples and label masks; verify only LoRA parameters are trainable.
- Pass requires 10 completed steps, finite loss and gradient norms, at least one nonzero adapter gradient, and changed adapter weights.
- Record elapsed time, peak GPU memory, trainable parameter count, and final loss in a machine-readable summary.
- Reload the saved adapter with the same base-model revision and template in a fresh process; require finite forward loss and a short generation.
- On OOM, retry once at **1,024 tokens**, regenerating eligible excerpts and recording the changed configuration. Report other failures explicitly without silently changing model or framework.

## Assumptions and current status

This establishes installation, training, and adapter usability—not improved terminal-task performance. Full trajectories, benchmark execution, and training-quality evaluation are outside this smoke test. Initial download time will likely dominate the run; reuse the scratch Hugging Face cache on subsequent runs.

Implemented and validated on November 23, 2026. The implementation-time GPU check found an idle B300 with all approximately **269 GiB VRAM** available.

- Created `setup.sh`, `smoke_test.py`, and `smoke_lora.yaml`. Run `./setup.sh` to install/check the environment, then `./setup.sh --run all` for a new run. Individual `prepare`, `train`, and `verify` commands accept `--root`, `--examples`, `--steps`, and `--seq-len`. Existing saved adapters are protected against overwrite; use a new scratch root for another training run.
- Installed managed Python **3.12.12**, the requested CUDA 13.0 PyTorch wheels, and Transformers **5.17.0**. LLaMA-Factory commit: `8ebba0d07dd14fdb1d429c497f018c7dae2e44b7`. Dependency/CUDA checks and a second idempotent setup run passed. The resolved lock is in scratch at `requirements.lock`.
- Model revision: `37561816e66b5b9153b1b64c10968c18c1e6f4a0`. Dataset revision: `45f1e98e317d3639437f0eb02f4e8bc5825732f3`.
- Prepared **32** complete prefixes, ranging from **1,497 to 2,032 tokens**, with exact template and user/padding loss-mask checks. The training dataset matched the inspected tokens and labels.
- Completed **10 optimizer steps** at the original **2,048-token** cutoff in **33.38 seconds** of trainer runtime. All losses and gradient norms were finite; nonzero adapter gradients and **992 changed adapter tensors** were confirmed. There were **58,363,904** trainable LoRA parameters. Final step loss: **0.875989**. Peak allocated GPU memory: **58.76 GiB**; peak reserved: **71.87 GiB**. No OOM retry was needed.
- Fresh-process verification reloaded the saved tokenizer and attached the adapter without merging. Forward loss: **0.655766**; a **16-token** greedy generation succeeded. The GPU was released after verification.
- Initial training exposed an upstream checkpointing signature mismatch: Transformers 5.17 passes `every_n_layers=1` and `offload=False`, which the pinned factory wrapper does not accept. `smoke_test.py` includes an explicit runtime compatibility wrapper accepting those default values while preserving the factory's checkpointing behavior; nondefault settings are rejected. No dependency constraints or package source files were changed. A subsequent verification argument-parsing issue was fixed by setting `do_train=False` for tokenizer setup and verification. Failure details and logs are retained in scratch.

All large artifacts are under `/scratch1/mendeza/qwen38-27b-llamafactory-test/`. The adapter and tokenizer are in `adapters/lora-2048/`; the final machine-readable result is `summary.json`; source indices and token counts are in `data/manifest.json`; formatted text and label masks are in `data/inspections.json`.
