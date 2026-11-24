# Terminal-Bench 2.1: base versus checkpoint 190

This is an exploratory five-task pilot, with one scored attempt per model and task.
The approved order is `fix-git`, `cancel-async-tasks`, `db-wal-recovery`,
`log-summary-date-ranges`, `build-cython-ext`. Base goes first on tasks 1, 3 and 5;
SFT goes first on tasks 2 and 4. There are ten planned scored trials and ten
additional calibration trials (oracle and no-op in fresh containers).

The full comparison is **not launched by setup or tests**. The existing SFT test
must finish before evaluation model preparation; the runner refuses an occupied
GPU and never kills an unrelated job.

## Current host validation (2026-11-24)

The repaired CLI preflight and fresh `fix-git` controls pass: oracle reward 1,
no-op reward 0. Exact rendered IDs match all four saved probes after normalizing
Harbor's reasoning field for both vLLM HTTP routes. Matching precompiled GPU
kernels are installed.

The user has explicitly authorized evaluation with numerical equivalence checks
treated as advisory. Their historical thresholds (merge 0.01, serving 0.02) and
observed differences remain recorded in provenance, local diagnostics and the
final report as future investigation. They do not block scoring. Exact adapter
loading, model/checkpoint integrity, tokenizer agreement and calibration remain
mandatory. The following diagnostics explain this documented limitation.

The base model's mean absolute chosen-token log-probability difference is
0.1494449 under default vLLM execution, above the historical 0.02 tolerance.
An eager-execution diagnostic reduced
it to 0.1043103 and still failed. Eager execution was not adopted in the runner,
and the historical tolerance remains unchanged. No scored model trial was
launched during these diagnostics. Evidence is saved locally in
`$SMOKE_ROOT/eval-implementation-validation/diagnosis.json`.

The three-way check also found that the old preparation did not load the trained
adapter: the multimodal checkpoint's `model.language_model` prefix differs from
the text-only evaluation class's `model` prefix. Its merged copy and reference
values are invalid for SFT scoring. Preparation now maps that prefix explicitly,
checks every adapter tensor against the checkpoint, and rejects any missing,
extra, or unequal tensor. Serving rejects old copies without this verification.
The original checkpoint is unchanged. New preparation records
`adapter-loading.json` and `merge-equivalence.json`, including failed equivalence
checks, so initialized adapter modules cannot masquerade as loaded weights.

For the valid base weights, the mean absolute differences were:

| Comparison | Difference |
| --- | ---: |
| Transformers 5.17 / Torch 2.10 versus Transformers 5.22.2 / Torch 2.14 | 0.0805568 |
| Original Transformers reference versus vLLM 0.34 | 0.1494449 |
| Transformers versus vLLM on the same serving library stack | 0.1394953 |

Thus matching the library stack alone does not meet the historical 0.02 threshold. Both servers
fit concurrently, and both real chat routes matched the tokenizer endpoint's
89-token count for the preserved-reasoning probe. Those checks used the old SFT
copy and establish runtime mechanics only. Detailed evidence and probe scripts
are in `$SMOKE_ROOT/eval-implementation-validation/three-way/`.

The corrected full-size preparation verified all 992 saved adapter tensors and
produced different SFT probe values, but failed the approved merge-equivalence
gate: mean chosen-token NLL difference 0.0357390 versus tolerance 0.01. A separate
diagnostic accumulated the delta in FP32 before rounding the final weights to
BF16 once; its difference was 0.0311202, also failing. That alternative was not
adopted. No corrected merged copy was published and no model trial was scored
during those diagnostics.
The evidence is `three-way/fixed-preparation/{adapter-loading,merge-equivalence}.json`
and `three-way/merge-single-rounding.json`. The subsequent user instruction
supersedes the original plan's numerical blocking policy. The runner continues
after recording these differences, retaining the existing BF16 merge, vLLM
version, model settings, task order and time budget.

## Current work and remaining work

The five-task pilot `terminalbench21-pilot-20261124T194005Z` passed preflight for
all five task images. Fresh `fix-git` controls returned oracle reward 1 and no-op
reward 0. The user requested shutdown during the `cancel-async-tasks` oracle
control, before model preparation or scoring. That control is recorded as
interrupted; **zero scored model attempts ran**. Evaluation processes were stopped
and the GPU was released. Local run artifacts are retained under
`$SMOKE_ROOT/evals/`; they are excluded from Git.

Next steps:

- Launch the full manual command below using a **new run directory**. Do not
  resume the stopped pilot: its interrupted control is retained and not retried.
- Complete the remaining task controls, rebuild the corrected BF16 SFT copy,
  and collect the ten official verifier scores. Numerical advisories should not
  block this exploratory run.
- Inspect infrastructure failures, missing rewards, agent behavior and paired
  outcomes before drawing conclusions. Publish only allowlisted aggregates via
  the exporter; transcripts remain local unless explicitly requested.
- Investigate BF16 merge rounding and differences between Transformers and vLLM
  in future work. Preserve probe values and package versions, distinguish library
  effects from kernel effects, and do not treat the historical invalid SFT copy
  as evidence about the trained adapter. Revisit numerical acceptance criteria
  separately from the current exploratory benchmark.

The reusable SFT workflow, progress reporting and isolated W&B integration are
documented in [SFT.md](SFT.md). Historical overnight entrypoints remain aliases.

## Install isolated dependencies

```bash
export SMOKE_ROOT="${SMOKE_ROOT:-/scratch1/$(id -un)/qwen38-27b-llamafactory-test}"
./setup.sh --setup-eval
# Optional, for isolated W&B aggregate reporting:
./setup.sh --setup-tracking
```

Harbor and vLLM each have a separate fully resolved lock file and virtual
environment. Training and tracking environments are preserved. Harbor is pinned
to `39c2219a41b3caa7a689cef1882263b5a93ae93f`; vLLM is `0.34.0`.
The vLLM lock also includes matching FlashInfer 0.7.3 cubin and CUDA 13.0
precompiled cache wheels from the vendor wheel index, so this host does not need
`nvcc` to warm up attention kernels. Setup checks their exact versions.
The pinned vLLM dependency graph requires newer Transformers/Hugging Face Hub
versions than training. Merge and reference inference therefore use the existing
training interpreter (Transformers `5.17.0`, PEFT `0.18.1`), without changing it.
There is no fallback to another engine, model, or precision.

Setup verifies the Harbor VCS installation metadata and vLLM version, checks
package consistency, and writes lock-digest readiness markers. Apptainer must be
available as `singularity` or `apptainer` on PATH; setup creates an evaluation-only
compatibility symlink when necessary. No Docker daemon is needed. Installation
time is outside the experiment budget.

## Container calibration without model preparation

Use a new run directory. The clock starts at the first invocation, and all later
phases share its absolute deadline. This command downloads the five pinned task
folders, resolves immutable Linux amd64 image digests, pulls their SIF images,
checks containment, verifies the original checkpoint, and audits instruction
overlap. It performs no model inference.

```bash
SFT_RUN="$SMOKE_ROOT/runs/overnight-20261124T022453Z"
EVAL_RUN="$SMOKE_ROOT/evals/terminalbench21-$(date -u +%Y%m%dT%H%M%SZ)"
./setup.sh --run terminalbench preflight \
  --sft-run "$SFT_RUN" --run-dir "$EVAL_RUN" --hours 3
./setup.sh --run terminalbench calibrate --resume \
  --sft-run "$SFT_RUN" --run-dir "$EVAL_RUN"
```

Each task must produce official oracle reward `1` and no-op reward `0` with no
Harbor exception. A missing reward or failed control blocks scoring; controls are
recorded separately and never counted as model attempts. An interrupted control
is retained and blocks that run rather than being repeated automatically.

Task files come from `harborframework/terminal-bench-2.1`, version `2.1.0`, revision
`3e235dff6880252a587fa479c09bcb1e16edf2eb`. `subset.json` records every original
file hash, complete task declarations, resolved OCI digests, SIF hashes, and hashes
of the effective local task copy. Only the image reference changes in that copy:
it points at a derived SIF that adds empty Harbor mountpoints. The original
SIF, derivation recipe, and derived SIF are separately hashed; task payload files
are unchanged. Agent and verifier timeouts remain 900 seconds.
Automatic Harbor trial/job reruns are not used.

The overlap audit extracts task descriptions from saved train, validation and
anchor prompts, excluding transient terminal screens and shared agent policy.
It reports normalized exact matches and potential substring/word-set matches
(Jaccard threshold 0.5). Exact matches block the selected subset; tasks are never
silently substituted. This check cannot exclude broader contamination.

## Single paired-task integration check

This runs two controls and two scored attempts for `fix-git`, including actual
GPU merge/reload checks and serving parity. Run it after the training GPU is free.
It uses its own directory and is explicitly marked as an integration check in
`summary.json`; five-task denominators remain visible.

```bash
./setup.sh --run terminalbench \
  --sft-run "$SMOKE_ROOT/runs/overnight-20261124T022453Z" \
  --run-dir "$SMOKE_ROOT/evals/terminalbench21-integration-$(date -u +%Y%m%dT%H%M%SZ)" \
  --tasks fix-git --hours 1
```

## Full approved configuration

```bash
./setup.sh --run terminalbench \
  --sft-run "$SMOKE_ROOT/runs/overnight-20261124T022453Z" \
  --run-dir "$SMOKE_ROOT/evals/terminalbench21-$(date -u +%Y%m%dT%H%M%SZ)" \
  --hours 3 --wandb-mode auto
```

Omit `--wandb-mode auto` to keep reporting entirely local. `offline` and `online`
are also supported. W&B runs only in the tracking interpreter with bounded SDK
initialization/finish settings and a supervisor heartbeat. A stalled reporter is
terminated independently; local journals remain available. W&B receives fixed,
allowlisted settings and aggregate numeric metrics, never arbitrary provenance,
paths, credentials, task instructions or transcripts. Offline W&B directories
under `tracking/` can be synced explicitly with the tracking environment's
`wandb sync` command.

Model preparation requires the original run's recorded selection to point at
checkpoint 190, validates its completion marker and file hashes, checks the
training identity/source/dependency lock, and verifies base-cache files against
Hugging Face metadata for the pinned model revision. The original adapter and
base cache are read only. A separate BF16 model is written to `merged-bf16/`,
reloaded, and checked against the in-memory merge. The mean absolute per-probe
chosen-token NLL difference from the unmerged adapter must be at most 0.01.
Fixed probes are short and unrelated to benchmark tasks; one includes prior
reasoning to check template preservation.

Both vLLM servers use the checkpoint's saved tokenizer/chat template, thinking
enabled, `xhigh` effort, preserved reasoning, greedy decoding, temperature 0,
top-p 1, seed 42, BF16, 65,536 context tokens, one active sequence, and GPU-memory
utilization 0.40 each. Servers bind to loopback and start sequentially; both must
fit before scoring. Rendered token IDs must match Transformers and mean absolute
chosen-token logprob difference must be at most 0.02 for each model. These gates
fail clearly rather than changing settings. **64K evaluation extends beyond the
adapter's 32K SFT window.**

`terminalbench_qwen.py` subclasses the pinned Terminus-2 agent. It keeps the
native JSON parser and terminal policy. It counts prompts with `/tokenize` using
the same messages and template options as generation and requires agreement with
the completion response's prompt usage. Native context summarization activates
when fewer than 33,024 tokens remain; the outgoing user prompt is also checked.
Every completion, including summary/question/answer/fallback calls, is bounded by
`min(32768, 65536 - rendered_prompt_tokens - 256)`. Effective limits, usage,
finish reasons, truncation and latency are recorded. Reasoning remains separate
from action JSON and is preserved in subsequent chat messages and local ATIF.
The transport normalizes Harbor's legacy `reasoning_content` to `reasoning` for
both HTTP routes because vLLM's tokenizer route does not apply the chat route's
legacy-field conversion. A short real chat probe also checks prompt-count
agreement before scoring.

## Phases, deadlines and recovery

The default `all` workflow calibrates containers before model preparation so
infrastructure failures are caught before a costly merge.

The supported phases are `preflight`, `prepare`, `calibrate`, `run`, `report`, and
`all` (the default). To continue the calibrated example above:

```bash
./setup.sh --run terminalbench prepare --resume \
  --sft-run "$SFT_RUN" --run-dir "$EVAL_RUN"
./setup.sh --run terminalbench run --resume \
  --sft-run "$SFT_RUN" --run-dir "$EVAL_RUN"
./setup.sh --run terminalbench report \
  --sft-run "$SFT_RUN" --run-dir "$EVAL_RUN"
```

Repeat `--tasks fix-git` when resuming/reporting an integration run. Resume cannot
extend the persisted deadline, change the selected run/task order, or repeat a
started attempt. Already completed preflight/preparation/calibration work is
reused with integrity checks. An interrupted attempt remains explicitly marked;
only unstarted model trials can run. Source, task, model, checkpoint, lock, and
description changes invalidate the run. A partially written merged copy is not
overwritten automatically; investigate the logs and use a fresh run directory.

The outer supervisor bounds even stalled library/network calls. At deadline
minus 60 seconds it requests shutdown, allows brief worker cleanup, then kills
only the private process groups created for this evaluation and writes reports.
Interrupted agent/verifier work and missing rewards remain explicit. Process
ownership uses a local journal, Linux start ticks, group identity and a run
marker before cleanup on resume; unrelated training processes are never targets.

## Container semantics and local artifacts

Harbor's built-in Singularity backend uses fakeroot and PID containment.
Apptainer extracts each SIF to a fresh writable temporary sandbox
(`APPTAINER_UNSQUASH=1`, `APPTAINER_WRITABLE=1`). This avoids rootless FUSE
permission errors and dpkg cross-device directory renames in writable overlays.
The original SIF remains unchanged. A derived SIF adds only empty `/staging`,
log mountpoints and the apt configuration mountpoint needed for writable
execution; its recipe and hash are recorded alongside the original image. Home, host temporary directories, configured bind
paths, current-directory mounts and host-filesystem mounts are suppressed. The
worker uses a restricted host environment and an empty home. Each trial gets
its own disposable copies of DNS settings and a rootless apt configuration,
bound to `/etc/resolv.conf` and `/etc/apt/apt.conf.d/99harbor-rootless`.
Agent, verifier and artifact output directories are also explicitly mounted from
the current trial only. This lets Harbor read official reward files directly and
avoids its staging-directory download path. These
contain no credentials. The pinned backend ignores per-bind `read_only` flags,
so only these disposable files are exposed, never the original host files. Repository, model,
original-run directories and host credentials are not mounted. Generated commands
run through Harbor's container terminal, never a host shell.

Resource declarations are retained. Unlike Docker, this backend does not enforce
CPU, memory and storage through cgroups. Harbor's native memory watchdog polls
process-tree PSS/RSS and kills at 95% of the declared memory; CPU and storage limits
are not enforced. Networking is shared with the host. These differences are
recorded in provenance and reports; this pilot makes no leaderboard-equivalence
or statistical-significance claim.

Local outputs include:

- `report.md`, `summary.json`, `trials.csv`, `state.json`, `progress.json`.
- `subset.json`, `provenance.json`, `overlaps.json`, `prepared.json`, reference
  probe values and serving-parity records.
- Harbor trial results, verifier logs, ATIF trajectories, inference/terminal
  timing journals, generated trial configurations and worker/server logs.
- Structured local journals in `events/`, plus optional isolated `tracking/` data.

Scores use official recorded verifier rewards, independently of model completion
claims. Agent errors/timeouts and Harbor's `reward_valid` flag remain visible
alongside rewards. Missing/nonfinite rewards stay unscored. Reports show solved
counts against five planned tasks per model, scored coverage out of ten and
paired wins/losses/ties only on tasks with two recorded rewards. `trials.csv`
includes prompt/completion tokens, turns, JSON parse failures, truncations,
inference/terminal/agent/verifier time, and infrastructure failures. Full
transcripts remain local.

Export sanitized aggregates only:

```bash
"$SMOKE_ROOT/eval-harbor-venv/bin/python" export_terminalbench_results.py \
  --run-dir "$EVAL_RUN" --output-dir results
```

## CPU validation

```bash
"$SMOKE_ROOT/eval-harbor-venv/bin/python" -m unittest test_terminalbench -v
bash -n setup.sh
./setup.sh --run terminalbench --help
```

Tests use mocked loopback endpoints and simulated Harbor results. They exercise
routing, reasoning separation/preservation, exact completion budgets (including
summary calls), pairing, calibration, missing rewards and agent errors, resume,
deadline shutdown, stalled workers/reporters, native Harbor construction, and
sanitized exports. They allocate no GPU and launch no benchmark comparison.
Actual adapter loading, serialized-copy integrity, dual-server fit, tokenizer
agreement and container controls must pass on the host before model scoring.
Merge and serving numerical differences are recorded as advisory diagnostics;
unit tests do not establish those runtime properties.
