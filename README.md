# Qwen3.8-27B LoRA training proof of concept

A reproducible, single-GPU [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) workflow for fine-tuning [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) on text trajectories from [TerminalHorizon-3K](https://huggingface.co/datasets/shunzou05/TerminalHorizon-3K).

**Measured on one NVIDIA B300:** the 2K smoke test passed training, adapter saving, and fresh-process reload. A 32K forward/backward/optimizer capacity probe also passed. The longer learning pilot was deliberately stopped after step 9 of 32; its step-8 checkpoint reduced held-out loss by **4.96%**. The 32K pilot has **not** completed final training or reload verification.

## Results

Recorded November 23, 2026. Small machine-readable evidence is committed under [`results/`](results/); weights, checkpoints, and dataset text are not included.

### Completed 2K smoke test

| Measurement | Result |
| --- | --- |
| Prepared dataset | 32 intact excerpts, at most 2,048 formatted tokens |
| Optimizer steps | 10 / 10 |
| Trainer runtime | 33.38 seconds |
| Trainable LoRA parameters | 58,363,904 |
| Mean training loss / final-step loss | 0.828033 / 0.875989 |
| Peak allocated / reserved GPU memory | 58.76 / 71.87 GiB |
| Gradient and update checks | Finite losses/norms; nonzero adapter gradients; 992 adapter tensors changed |
| Fresh-process verification | Passed: forward loss 0.655766 and 16 generated tokens |

This smoke test establishes that training and the saved adapter work. Its reload loss is on a prepared training example, **not a held-out quality measurement**. The 10 steps visit a subset of the 32-example dataset. [Full smoke-test metrics](results/smoke_test.json).

### 32K capacity probe and partial learning pilot

| Measurement | Result |
| --- | --- |
| Context cutoff | 32,768 tokens; no 16K fallback needed |
| Capacity-probe example | 32,759 tokens |
| Capacity check | Two accumulated backward passes and one optimizer step passed |
| Probe peak allocated / reserved GPU memory | 179.10 / 197.47 GiB |
| Pilot split | 64 training / 16 validation trajectories from 80 distinct task groups |
| Prepared training lengths | 20,517–32,759 tokens, 12–44 messages |
| Prepared validation lengths | 24,575–32,241 tokens, 10–32 messages |
| Planned / completed / saved steps | 32 planned; 9 completed; checkpoint 8 saved |
| Base-model validation loss | 1.077827 |
| Adapter validation loss, step 8 | 1.024374 |
| Relative held-out loss reduction | **4.96%** |
| Stop reason | User requested stopping and moving execution to another terminal |
| Final 32K adapter verification | Not yet run |

Validation loss is the mean of per-example assistant-token cross entropy over the same 16 held-out examples. The observed reduction is just below the pilot's predefined **5%** learning-signal threshold. It is a partial-run result, not evidence that the full pilot passed or that terminal-task completion improved. No commands from the dataset or model generations are executed.

See [pilot evidence](results/pilot_32k.json), [training and validation curves](results/training_curves.csv), and [hardware/runtime versions](results/environment.json). Capacity-probe memory is measured separately from the subsequent training run.

## Reproduce

This was tested on Linux with one B300 (approximately 269 GiB VRAM). Other hardware is untested. Use a fast scratch filesystem with room for roughly 56 GB of model weights plus the environment, caches, data, and checkpoints. The scripts check CUDA, BF16, a CUDA matrix multiplication, free GPU memory, and package consistency.

```bash
git clone https://github.com/interactivetech/qwen38-27b-llamafactory-poc.git
cd qwen38-27b-llamafactory-poc

# Set this to your own fast scratch directory before installing or running.
export SMOKE_ROOT=/path/to/fast-scratch/qwen38-27b-llamafactory-test
./setup.sh

# Original installation/training/save/reload smoke test.
./setup.sh --run all

# New 32K learning pilot; retry once at 16K only on CUDA OOM.
./setup.sh --run pilot --seq-len 32768 --fallback-seq-len 16384 \
  --examples 64 --eval-examples 16
```

Each pilot creates a timestamped run directory under `$SMOKE_ROOT/runs/`. The original smoke adapter is protected against overwrite. For another independent smoke run, use another scratch root; do not delete a saved adapter to rerun it. No external experiment reporting, quantization, DeepSpeed, optional attention kernels, or full-model merging is enabled.

The full 32K pilot was estimated at roughly 2–2.5 hours plus generation checks using the measured worst-case probe on this machine. This is an estimate, not a completed-run timing. Reference PyTorch kernels are intentionally used and are slow at long context lengths.

### Resume a stopped pilot

Data, the base-model baseline, the capacity probe, and the checkpoint must already exist locally; they are not downloaded from this repository. Point `PILOT_RUN` at your run directory. Resume selects its latest saved checkpoint, continues to the original step budget, and automatically runs final adapter verification.

```bash
export PILOT_RUN="$SMOKE_ROOT/runs/pilot-<your-run-id>"
nohup ./setup.sh --run pilot --phase resume --run-dir "$PILOT_RUN" \
  > "$PILOT_RUN/resume.log" 2>&1 &
tail -f "$PILOT_RUN/resume.log"
```

The published partial run's ID is `pilot-20261123T191916743546Z`. It resumes from step 8; unsaved step 9 is repeated. Press Ctrl-C to stop following the log without stopping the `nohup` job. A resumed run records errors and stops; it does not silently restart training at a different context length. Use `--seq-len 16384` when resuming a run that had already fallen back to 16K.

## Method and pinned stack

- **Data:** stream the pinned source in stable order; deduplicate and retain one trajectory per normalized initial task prompt, excluding its volatile terminal-state section. Seed 42 assigns 80 eligible task groups to the two disjoint splits. A fallback keeps those same source trajectories and split assignments.
- **Formatting:** longest intact initial conversation prefix ending in an assistant turn. Preserve reasoning, JSON commands, observations, and roles. Use `qwen3_8` and ShareGPT `role`/`content` fields; mask user messages and padding. Tokenized data is checked against the template before training.
- **Pilot training:** fresh BF16 base model, LoRA rank 8 / alpha 16 / dropout 0; supported language-model linear layers only. Base and vision weights stay frozen. Microbatch 1, accumulation 2, gradient checkpointing, no packing, AdamW at 1e-4 with a constant schedule, 32 optimizer steps / one epoch.
- **Evaluation:** unadapted baseline on 16 held-out examples; validation and resumable checkpoints every 8 steps; keep two checkpoints. Final verification reloads the saved tokenizer and unmerged adapter in a fresh process, repeats held-out loss, and compares four fixed base/adapter generations, capped at 256 new tokens each.

| Component | Pinned revision/version |
| --- | --- |
| Model revision | `37561816e66b5b9153b1b64c10968c18c1e6f4a0` |
| Dataset revision | `45f1e98e317d3639437f0eb02f4e8bc5825732f3` |
| LLaMA-Factory commit | `8ebba0d07dd14fdb1d429c497f018c7dae2e44b7` |
| Managed Python | 3.12.12 |
| PyTorch / torchvision / torchaudio | 2.10.0 / 0.25.0 / 2.10.0, CUDA 13.0 wheels |
| Transformers | 5.17.0 |
| Attention | PyTorch SDPA and standard model kernels |

The complete dependency resolution is in [`requirements.lock`](requirements.lock), with LLaMA-Factory pinned to its Git commit. Setup uses that lock for a fresh scratch environment and retains an existing scratch lock when rerun.

The pinned Factory checkpointing wrapper does not accept the new default `every_n_layers=1` / `offload=False` arguments passed by Transformers 5.17. The runner applies an explicit runtime compatibility wrapper for those defaults and rejects nondefault settings. Package source files and dependency constraints are not modified.

## Files, checks, and updating results

- [`setup.sh`](setup.sh): scratch environment, dependency lock, and CUDA checks.
- [`smoke_test.py`](smoke_test.py), [`smoke_lora.yaml`](smoke_lora.yaml): smoke workflow and shared training defaults.
- [`pilot_test.py`](pilot_test.py): split preparation, capacity probe, baseline, training, resume, and verification.
- [`test_pilot.py`](test_pilot.py): CPU tests for task identity, intact prefixes, OOM fallback, and checkpoint resume orchestration. These do not run GPU training.
- [`PILOT.md`](PILOT.md) and [`PLAN.md`](PLAN.md): detailed workflow and original experiment history. Original machine paths in the historical notes are examples; set `SMOKE_ROOT` for your system.

Run the CPU tests using the installed environment:

```bash
TMPDIR="$SMOKE_ROOT/tmp" PYTHONPYCACHEPREFIX="$SMOKE_ROOT/cache/pycache" \
  "$SMOKE_ROOT/venv/bin/python" -m unittest -v test_pilot
```

After continuing training, refresh the committed evidence from your actual run artifacts:

```bash
"$SMOKE_ROOT/venv/bin/python" export_results.py \
  --root "$SMOKE_ROOT" --pilot-run "$PILOT_RUN" --output results
```

The exporter allowlists metrics and omits dataset text, generated responses, absolute paths, and model/checkpoint files. If a final report exists, the JSON snapshot records the completed result; until then, it explicitly records a partial run. Update the README results table to match the new snapshot before publishing another commit.

This is an installation and training feasibility proof of concept with exploratory held-out loss measurements. It does not include an executable terminal-task benchmark. The upstream model, dataset, and framework remain separate projects with their own terms and licenses.
