# Held-out long-context LoRA pilot

Run from this workspace after the existing environment has been installed:

```bash
./setup.sh --run pilot --seq-len 32768 --fallback-seq-len 16384 --examples 64 --eval-examples 16
```

The runner reuses the pinned environment and Hugging Face caches under the existing scratch root. Each invocation creates a separate timestamped directory under `runs/`; the original smoke-test adapter and summary are preserved. `--run-dir` can specify a new directory explicitly. The `--root` option selects an already installed environment/cache root.

The selected dataset trajectories are streamed in stable source order, deduplicated, and limited to one trajectory per task group. Task identity hashes the whitespace-normalized initial prompt before its terminal-state section, so changing container IDs cannot place the same task in both splits. Seed 42 shuffles 80 eligible task groups into 64 training and 16 validation examples. Selection is saved before training. The longest complete initial prefix fitting the formatted cutoff is retained for each selected trajectory; reasoning, JSON commands, roles, and observations remain intact. The factory's processor produces labels, which are checked against the template's expected assistant-only supervision. The registered ShareGPT source files, tokenized Arrow datasets, source indices, hashes, example lengths, and formatted inspections are retained in the run directory.

Execution proceeds through separate processes for preflight, preparation, capacity probe, base-model baseline, training, and final verification. The probe uses the longest prepared example, two accumulated backward passes, and an AdamW step. It records peak allocated/reserved GPU memory and estimates training/evaluation runtime; generation time is excluded from that estimate. Probe weights and optimizer state disappear with that process. The learning run creates a new adapter from the base revision, with rank 8, alpha 16, BF16 base weights, SDPA, checkpointing, learning rate 1e-4, a constant schedule, microbatch 1, and accumulation 2. The default 32 optimizer steps consume exactly one epoch. All non-LoRA and vision parameters stay frozen.

The unadapted base model is scored on the 16 held-out examples before training. Training evaluates them every 8 steps and saves full resumable adapter checkpoints, including optimizer, scheduler, and RNG state; the latest two checkpoints are retained. The final adapter and tokenizer are saved separately. Final verification reloads them in a fresh process, repeats the held-out scoring, and compares the saved loss against the training evaluation. Base and adapter each generate up to 256 tokens greedily on four identical held-out prompts. These outputs are saved for inspection; commands are never executed.

On a CUDA OOM, the process exits and the complete pipeline restarts once at 16K using the same source trajectories and split assignments. Prefixes and the baseline are regenerated; learning restarts from the base model. No unrelated GPU process is stopped. A second OOM or any other error stops the run and records the failure. A process exit without a Python OOM report is not silently treated as OOM.

`run_status.json` identifies the current phase. During training, `context-<cutoff>/progress.json` gives completed steps and validation measurements. Each phase has a separate log and failures have structured tracebacks. The run's final `summary.json` includes revisions, dependency lock hash, example identities, capacity results, training/validation curves, memory and timing, fallback history, checkpoint paths, and generation comparisons.

`training_success` requires finite loss/gradient norms, 32 completed steps, nonzero adapter gradients, changed adapter tensors, saved resumable checkpoints, and successful fresh-process verification. `learning_signal` additionally requires at least a 5% reduction in the mean per-example assistant-token validation cross entropy versus the unadapted base model. With only 16 held-out tasks, this is an exploratory learning signal; it does not establish improved terminal-task completion. Generation outputs may reach the token cap during reasoning.

Focused runner tests can be executed with the installed Python:

```bash
TMPDIR=/scratch1/mendeza/qwen38-27b-llamafactory-test/tmp \
PYTHONPYCACHEPREFIX=/scratch1/mendeza/qwen38-27b-llamafactory-test/cache/pycache \
/scratch1/mendeza/qwen38-27b-llamafactory-test/venv/bin/python -m unittest -v test_pilot
```

Implementation run: `/scratch1/mendeza/qwen38-27b-llamafactory-test/runs/pilot-20261123T191916743546Z/`.

Training was stopped at the user's request after step 9. Checkpoint 8 was saved at that stop; its held-out loss was **1.024374**, compared with **1.077827** for the base model (a **4.96%** reduction). The 32K probe passed with **179.1 GiB** peak allocated memory; no context fallback was needed.

Resume that run from another terminal and automatically verify the final adapter:

```bash
cd /nfshomes/mendeza/2026_projects/2026_openagents_sft_test/qwen38-27b-llamafactory-test
PILOT_RUN=/scratch1/mendeza/qwen38-27b-llamafactory-test/runs/pilot-20261123T191916743546Z
nohup ./setup.sh --run pilot --phase resume --run-dir "$PILOT_RUN" \
  > "$PILOT_RUN/resume.log" 2>&1 &
tail -f "$PILOT_RUN/resume.log"
```

`resume` selects the latest complete checkpoint in the selected context directory. It validates the saved step budget, restores historical training/evaluation metrics, and runs only training and final verification. Data preparation, the capacity probe, and baseline evaluation are reused. An optional `--resume-from-checkpoint` selects a checkpoint explicitly. This continuation stops and records errors; it does not automatically start a new fallback run. Pressing Ctrl-C while following the log stops the log viewer; the `nohup` job continues. The final report will be written to `$PILOT_RUN/summary.json`.

The pilot subsequently resumed from checkpoint 8. At the step-16 evaluation, validation loss was **1.008066**, a **6.47%** reduction from the base-model baseline; checkpoint 16 was saved. This exceeds the 5% learning-signal threshold at an intermediate checkpoint. Training is still in progress, and final verification has not yet run. The committed result snapshots and README reflect this update.
