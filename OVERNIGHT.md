# Overnight 32K experiment

The overnight runner is separate from the existing pilot. Its defaults implement the approved experiment: fresh rank-8 LoRA from the pinned base, 400 training groups (including the original 64), 32 new validation groups, 32 new test groups, and the original 16 validation groups held separately as anchors. Complete conversation prefixes, reasoning, observations, and assistant-only labels are retained at 32,768 tokens. Original pilot rows are checked against their token hashes.

The supervisor records absolute deadlines before preflight. A 12-hour run stops training at an optimizer boundary at 200 steps or 10.5 hours from launch, reserving 90 minutes for stop-time validation and final comparisons. Preparation and the disposable longest-example probe count toward this budget. A slower run can complete fewer than 400 examples. The command does not alter context length or retry with different parameters after failure.

## Launch

Use the existing pinned environment; no setup or dependency update is needed if the pilot environment is intact.

```bash
cd /nfshomes/mendeza/2026_projects/2026_openagents_sft_test/qwen38-27b-llamafactory-test
export SMOKE_ROOT=/scratch1/mendeza/qwen38-27b-llamafactory-test
REFERENCE_RUN="$SMOKE_ROOT/runs/pilot-20261123T191916743546Z"
OVERNIGHT_RUN="$SMOKE_ROOT/runs/overnight-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OVERNIGHT_RUN"
nohup ./setup.sh --run overnight \
  --reference-run "$REFERENCE_RUN" --run-dir "$OVERNIGHT_RUN" \
  --hours 12 --seq-len 32768 \
  --train-examples 400 --val-examples 32 --test-examples 32 \
  > "$OVERNIGHT_RUN/overnight.log" 2>&1 &
tail -f "$OVERNIGHT_RUN/overnight.log"
```

Ctrl-C exits the log viewer. The supervisor owns only its spawned worker process groups; it does not stop unrelated GPU processes. It requests graceful shutdown with stop files, gives a training worker up to 120 seconds to finish an optimizer boundary/checkpoint, and sends TERM then KILL if needed. The worker cutoff reserves the last 15 seconds for CPU reporting; the supervisor requests shutdown 60 seconds before that cutoff, sends TERM with 10 seconds left, and enforces the worker cutoff with KILL. CPU reporting avoids reading checkpoint weight files again. A watchdog termination is reported as failure, preserving completed artifacts. Disk preflight requires 24 GiB free; approximately 17–20 GiB is expected for all 25 resumable checkpoints with this adapter.

`status.json` is refreshed every 30 seconds even if the model worker is stalled. `progress.json` records each optimizer step and evaluation/generation progress, including allocated and reserved CUDA memory separately. Worker logs live under `logs/`; `overnight.log` records phase changes. Inspect `progress.json` or the current worker log for detailed training progress.

## Inspect or resume

```bash
export SMOKE_ROOT=/scratch1/mendeza/qwen38-27b-llamafactory-test
OVERNIGHT_RUN=$(cat "$SMOKE_ROOT/latest_overnight_run.txt")
cat "$OVERNIGHT_RUN/status.json"
cat "$OVERNIGHT_RUN/report.md"
```

To resume an interrupted run, repeat its launch arguments with `--resume` and the **same run directory**. The original absolute deadlines, dependency lock, reference adapter hashes, revisions, and experiment settings must match. An advisory lock prevents two supervisors from running in the same directory. An expired run produces a partial report without launching model work. A completed training phase is not repeated.

Resume restores the latest completed checkpoint's adapter, AdamW state, constant scheduler, RNG states, counters, and per-step history. A deterministic seed-42 permutation and the checkpoint's example offset preserve the one-pass order. Steps after the latest saved checkpoint may need to be repeated. Incomplete temporary checkpoint directories are ignored; missing or corrupt published artifacts cannot be resumed. All completed checkpoints remain available. The selected adapter is used directly from its checkpoint and never overwrites the latest resumable checkpoint.

## Outputs and interpretation

`report.md` and `summary.json` include held-out and anchor losses, paired task-bootstrap 95% intervals, timings, memory, failures, and skipped work. `training_curves.csv` and `validation_curves.csv` contain the curves. `selected_checkpoint.json` records validation-only selection; `latest_checkpoint.json` records the latest resumable checkpoint. Each published checkpoint has a checksummed `complete.json`; only completed validation sidecars are eligible for selection.

Final scoring uses fresh processes for base, pilot, and selected adapter on identical examples. The selected adapter's reloaded validation loss must agree with checkpoint validation before it is marked verified. Losses are mean per-example assistant-token cross entropy, matching the pilot. The bootstrap resamples the same task indices for both models; intervals do not capture variability across training seeds.

`paired_responses.json` and `paired_responses.md` contain four new test-context generations and the four existing controlled instruction probes for each model. Each prompt reserves 2,048 output tokens within 32K, and generation is greedy. JSON parsing, schema completion, output-cap truncation, and exact controlled-instruction checks are reported. Test-context schema checks are not independent task-success judgments. Generated commands are never executed.

At least 5% lower test loss than base **and** lower test loss than pilot is labeled promising only when finite training metrics, nonzero adapter gradients, changed weights, and fresh-process reload have been verified. Neither that threshold nor eight static responses establishes terminal-task success. Unfinished evaluations are explicitly marked; per-example partial results remain on disk. Reports distinguish the latest recoverable checkpoint from last observed worker progress.

## Verification without overnight training

```bash
./setup.sh --run overnight --help
"$SMOKE_ROOT/venv/bin/python" -m unittest test_overnight test_pilot test_quality -v
```

Tests use temporary directories, CPU model steps, and simulated workers with a fake clock. They cover isolation, checksum completion markers, validation selection, resume history/deadlines, nonfinite gradients/losses, graceful and forced shutdown, and partial reporting. Full 27B capacity, reload, and evaluation verification occurs during the actual run, not in these tests.
