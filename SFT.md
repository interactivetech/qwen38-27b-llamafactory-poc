# SFT training

The supervised fine-tuning (SFT) runner supports bounded long-context LoRA experiments, live terminal progress, and local or online W&B tracking. It reuses the pinned model/environment and the pilot's data preparation and verification routines. The smoke and pilot commands retain their existing behavior.

The current experiment defaults remain 32,768 tokens, rank 8/alpha 16/dropout 0, BF16, SDPA, gradient checkpointing, microbatch 1, accumulation 2, AdamW, constant learning rate 1e-4, and seed 42. Training selects 400 independent task groups (including the original 64), with 32 new validation groups, 32 new test groups, and the original 16 validation groups as a separate anchor set. Complete prefixes, reasoning, observations, and assistant-only labels are preserved and hashed.

A 12-hour budget covers preparation, a disposable capacity probe, training, evaluation, and reporting. Training stops at an optimizer boundary after 200 steps or 10.5 hours from launch. Remaining time is reserved for validation, base/pilot/selected comparisons, and generations. A budget-limited run can process fewer examples. No automatic context-length or hyperparameter fallback occurs.

## Setup and launch

For a machine with the existing pilot environment, install the separate tracking environment once:

```bash
export SMOKE_ROOT=/scratch1/mendeza/qwen38-27b-llamafactory-test
./setup.sh --setup-tracking
```

Fresh `./setup.sh` setup provisions both environments. `requirements.lock` and the training environment are unchanged by tracking setup. W&B and its dependencies have their own lock and live in `$SMOKE_ROOT/tracking-venv`.

Launch a new SFT experiment:

```bash
export SMOKE_ROOT=/scratch1/mendeza/qwen38-27b-llamafactory-test
REFERENCE_RUN="$SMOKE_ROOT/runs/pilot-20261123T191916743546Z"
RUN_DIR="$SMOKE_ROOT/runs/sft-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_DIR"
nohup ./setup.sh --run sft \
  --reference-run "$REFERENCE_RUN" --run-dir "$RUN_DIR" \
  --hours 12 --seq-len 32768 \
  --train-examples 400 --val-examples 32 --test-examples 32 \
  > "$RUN_DIR/training.log" 2>&1 &
tail -f "$RUN_DIR/training.log"
```

For a short integration test, add `--max-steps 5 --train-examples 64 --val-examples 4 --test-examples 4 --hours 2`. This performs five real optimizer steps (ten training examples) and saves checkpoint 5, even though regular checkpointing happens every eight steps. Preparation, the capacity probe, stop-time validation, and the final evaluation/generation comparisons still run. `--max-steps` must be at least 1 and no greater than `min(200, train_examples / 2)`; keep the same value when resuming. To require an online attempt, use `--wandb-mode online`; connection failures still fall back locally as described below.

The log shows completed optimizer steps, loss, gradient norm before clipping, learning rate, epoch fraction, step time, tokens/second, tokens processed, allocated/reserved memory, and elapsed time. `target_eta~` estimates time to the requested step count using a trailing ten-step average; `training_budget_left` is the separate, fixed deadline. Estimates exclude future evaluation/checkpoint overhead. Evaluation progress names the model, split, checkpoint, and example count; generation progress shows response counts.

`--progress auto` (default) uses updating progress bars in an interactive terminal and newline-separated records in redirected logs. `--progress lines` always uses lines; `--progress off` suppresses progress rendering but retains local records and the final result. A heartbeat appears every 30 seconds during long operations, including model loading. Each run ends with an explicit `SFT COMPLETE`, `SFT PARTIAL`, or `SFT FAILED` message and report/tracking locations.

**`tail -f` continues waiting after the experiment finishes.** Press Ctrl-C to close the viewer; the background experiment is unaffected. Raw model-worker output remains in `logs/<phase>-<timestamp>.log`, whose location is printed when each phase starts.

## W&B modes

`--wandb-mode auto` is the default:

- Without a nonempty `WANDB_API_KEY`, W&B records **offline** in the local run directory. No online authentication is attempted.
- With `WANDB_API_KEY`, W&B attempts **online** tracking and prints the hosted run URL when ready.
- If the connection fails or the reporter stalls, training continues and local events are retained. Failed online initialization is replaced by a fresh offline reporter. Runtime outages remain buffered locally; failed shutdown/upload is marked for recovery.

Set a key in your shell using your normal credential workflow; never put it in a launch command that will be committed. Cached SDK login credentials can be used with explicit `--wandb-mode online`. `--wandb-mode offline` forces local recording even when a key exists; `--wandb-mode disabled` needs no tracking installation and still retains the local progress journals. No mode prompts for a login inside a job.

Project/entity/run-name precedence is CLI (`--wandb-project`, `--wandb-entity`, `--wandb-name`), then `WANDB_PROJECT`/`WANDB_ENTITY`/`WANDB_NAME`, then defaults. The default project is `qwen38-27b-llamafactory-poc`, the default run name is the run-directory basename, and the default entity is the authenticated account's SDK default.

The dashboard includes training and checkpoint-validation curves, pre-clipping gradient L2 norms, learning rate, token counts/throughput, memory, phase progress, budget remaining, and final loss/uncertainty/quality comparisons. Training and checkpoint-validation curves use optimizer-step axes; resource and phase metrics use elapsed time. Configuration is allowlisted. No model weights, prompt transcripts, generated responses, raw console logs, source files, or API keys are uploaded by this integration. W&B may retain SDK diagnostics and dependency-version metadata.

**Offline W&B recording does not provide a hosted live dashboard.** To sync local data later:

```bash
RUN_DIR=$(cat "$SMOKE_ROOT/latest_sft_run.txt")
./setup.sh --tracking sync --run-dir "$RUN_DIR"
```

Sync refuses a run whose supervisor is active. Failed/incomplete reporter sessions are reconstructed from the local journals before syncing; completed offline SDK files are synced directly. Already-synced segments are skipped. Reconstructed segments supersede incomplete local segments; an incomplete online segment may remain visible in W&B alongside its grouped recovery segment. A later explicit sync uses your W&B credentials and may require network access.

Tracking runs in a separate process and environment; SDK/network operations never run in the model workers or the deadline watchdog. Online startup is bounded to 15 seconds, and final flushing to at most five seconds within the reporting reserve. `tracking/status.json` records requested/effective mode, reporter health, pending sync, and URL/local directory. Reporter diagnostics live under `tracking/sessions/`.

## Inspect, resume, and compatibility

```bash
RUN_DIR=$(cat "$SMOKE_ROOT/latest_sft_run.txt")
cat "$RUN_DIR/status.json"
cat "$RUN_DIR/tracking/status.json"
cat "$RUN_DIR/report.md"
```

`status.json` is refreshed every 30 seconds; `progress.json` contains the current activity without stale fields from earlier phases. Detailed progress is preserved in append-only per-producer journals under `telemetry/<session>/`. A new tracking session is created per supervisor invocation, grouped under the same logical experiment. Repeated optimizer steps after checkpoint rollback belong to the new session; event IDs prevent duplicate replay within a reporter segment. Cursor files record consumed offsets and event IDs. Recovery uses a fresh SDK segment because offline SDK files do not support resume in place.

Resume with the **same launch arguments and run directory**, adding `--resume`. Adapter, optimizer, constant scheduler, RNG states, counters, and history come from the latest checksummed complete checkpoint. A deterministic permutation preserves example order. Steps after the latest checkpoint may repeat. Original absolute deadlines, revisions, dependency lock, and reference hashes remain enforced. Tracking settings are separate from training identity, so they can change on resume without changing the experiment.

The supervisor stops only its own worker process groups. It requests graceful shutdown with stop files, allows up to 120 seconds to finish an optimizer boundary/checkpoint, then uses TERM/KILL if needed. The worker cutoff reserves 15 seconds for reporting; graceful overall shutdown begins 60 seconds before that cutoff, followed by TERM with 10 seconds left and KILL at the cutoff. At least 24 GiB free disk space is required. All completed checkpoints remain available; approximately 17–20 GiB is expected for the default run.

`./setup.sh --run overnight`, `overnight.py`, and `export_overnight_results.py` are deprecated compatibility entrypoints. They print replacement instructions. Existing scratch directories, checkpoint identities, and historical `results/overnight_*` links are preserved. New exports use `sft_*`; the legacy exporter retains the old filename prefix. The supervisor updates both `latest_sft_run.txt` and the legacy pointer. Pointer-reading utilities prefer the new pointer and fall back to the legacy pointer. Before the first new SFT run, use `RUN_DIR=$(cat "$SMOKE_ROOT/latest_sft_run.txt" 2>/dev/null || cat "$SMOKE_ROOT/latest_overnight_run.txt")` when inspecting an existing experiment.

## Artifacts and interpretation

`report.md` and `summary.json` include losses, paired task-bootstrap 95% intervals, quality counts, timings, memory, failures, and skipped work. `training_curves.csv` and `validation_curves.csv` hold the curves. `selected_checkpoint.json` records validation-only selection; `latest_checkpoint.json` records the latest resumable checkpoint. Only checksummed checkpoints with completed validation sidecars can be selected.

Final scoring uses fresh processes and identical test examples for base, pilot, and selected models. The selected model's reloaded validation loss must agree with checkpoint validation. Losses are mean per-example assistant-token cross entropy. The bootstrap resamples paired task indices; it does not represent variability across training seeds.

Four new test-context generations and four controlled instruction probes per model are saved locally in paired response files. Greedy decoding reserves 2,048 output tokens inside the context budget. Generated commands are never executed. JSON/schema checks and static instruction probes do not establish terminal-task success. At least 5% lower test loss than base and lower test loss than pilot is promising only after finite metrics, nonzero gradients, changed weights, and reload verification pass.

Export metrics and PNG/SVG figures:

```bash
"$SMOKE_ROOT/venv/bin/python" export_sft_results.py --run-dir "$RUN_DIR" --output results
```

Add `--include-responses` only when explicitly publishing full response comparisons.

## CPU-only verification

```bash
"$SMOKE_ROOT/venv/bin/python" -m unittest test_sft test_sft_telemetry test_pilot test_quality
DEMO_RUN="$SMOKE_ROOT/runs/sft-demo-$(date -u +%Y%m%dT%H%M%SZ)"
nohup ./setup.sh --run sft-demo --run-dir "$DEMO_RUN" --wandb-mode offline \
  > "$SMOKE_ROOT/logs/sft-demo.log" 2>&1 &
tail -f "$SMOKE_ROOT/logs/sft-demo.log"
```

The demo runs three tiny CPU optimizer steps through the real supervisor/renderer/reporter and records offline W&B data. Its evaluation/generation counters are synthetic and clearly labeled as a demonstration. It does not load the 27B model, alter experiment pointers, or perform a training experiment. Full model training is not required to validate logging and tracking changes.
