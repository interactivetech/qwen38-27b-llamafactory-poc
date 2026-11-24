"""Reproducible, deadline-bounded Terminal-Bench 2.1 pilot supervisor (stdlib only)."""

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import tomllib
import unicodedata
import urllib.parse
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from sft_progress import emit
from terminalbench_qwen import CONTEXT, OUTPUT, SUMMARIZE, TEMPLATE_KWARGS, request

HARBOR = "39c2219a41b3caa7a689cef1882263b5a93ae93f"
DATASET = "harborframework/terminal-bench-2.1"
REVISION = "3e235dff6880252a587fa479c09bcb1e16edf2eb"
TASKS = [
    "fix-git",
    "cancel-async-tasks",
    "db-wal-recovery",
    "log-summary-date-ranges",
    "build-cython-ext",
]
REPO = Path(__file__).resolve().parent
RESOURCE_NOTE = (
    "Singularity retains task declarations but has no Docker cgroup CPU/memory/storage "
    "enforcement. Harbor polls process-tree PSS/RSS and kills at 95% of declared memory; "
    "CPU and disk limits are not enforced. Host networking is shared. Each SIF is extracted "
    "to a fresh writable temporary sandbox to support rootless dpkg operations. Disposable "
    "per-trial DNS and apt configuration copies are the only added mounts; Harbor's "
    "per-bind read_only flags are not honored by this pinned backend. Results are exploratory."
)


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def hashes(root, names=None):
    root = Path(root)
    return {
        str(p.relative_to(root)): sha(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and (names is None or str(p.relative_to(root)) in names)
    }


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def instruction_from_training(content):
    # Saved initial user prompts include the Terminus policy and an ephemeral terminal screen.
    prefix = content.split("Current terminal state:", 1)[0]
    for marker in ("Task Description:", "Task description:", "Task:\n"):
        if marker in prefix:
            return prefix.split(marker, 1)[1].strip()
    raise ValueError(
        "Unknown training instruction framing; cannot check overlap reliably"
    )


def overlaps(sft_run, instructions):
    selection = read(sft_run / "selected_trajectories.json")
    records = read(sft_run / "context-32768/data/manifest.json")["records"]
    included = {
        r["conversation_sha256"]: split
        for split, rows in records.items()
        if split in ("train", "validation", "anchor")
        for r in rows
    }
    saved = []
    for item in selection["trajectories"]:
        digest = item["conversation_sha256"]
        if digest in included:
            saved.append(
                (
                    included[digest],
                    digest,
                    normalized(
                        instruction_from_training(item["conversations"][0]["content"])
                    ),
                )
            )
    if len(saved) != len(included):
        raise ValueError("Saved training/validation descriptions are incomplete")
    result = dict(
        checked=len(saved),
        exact=[],
        potential=[],
        limitation="Normalized instruction comparison cannot exclude broader contamination.",
    )
    for name, instruction in instructions.items():
        instruction = normalized(instruction)
        words = set(instruction.split())
        for split, digest, text in saved:
            if instruction == text:
                result["exact"].append(
                    dict(task=name, split=split, conversation=digest)
                )
            else:
                similarity = len(words & set(text.split())) / max(
                    1, len(words | set(text.split()))
                )
                if instruction in text or text in instruction or similarity >= 0.5:
                    result["potential"].append(
                        dict(
                            task=name,
                            split=split,
                            conversation=digest,
                            word_jaccard=similarity,
                        )
                    )
    return result


def verify_checkpoint(sft_run):
    if sft_run.name != "overnight-20261124T022453Z":
        raise ValueError(
            "Use the explicitly approved original overnight-20261124T022453Z run"
        )
    chosen = read(sft_run / "selected_checkpoint.json")
    expected = sft_run / "context-32768/checkpoints/checkpoint-190"
    if (
        not chosen
        or chosen["step"] != 190
        or Path(chosen["path"]).resolve() != expected.resolve()
    ):
        raise ValueError("Original run selection is not checkpoint 190")
    marker = read(expected / "complete.json")
    if not marker or not marker.get("files"):
        raise ValueError("Checkpoint completion manifest missing")
    experiment = read(sft_run / "experiment.json")
    identity_hash = hashlib.sha256(
        json.dumps(experiment["identity"], ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    if marker["identity"] != identity_hash:
        raise ValueError("Checkpoint identity does not match original experiment")
    for name, digest in marker["files"].items():
        if Path(name).name != name or sha(expected / name) != digest:
            raise ValueError(f"Checkpoint integrity failure: {name}")
    if read(expected / "trainer_state.json")["global_step"] != 190:
        raise ValueError("Checkpoint state is not step 190")
    validation = read(expected / "validation.json")
    if not validation.get("complete") or validation.get("checkpoint_sha256") != sha(
        expected / "complete.json"
    ):
        raise ValueError("Checkpoint selection validation is incomplete or changed")
    return expected, experiment


def gpu_free():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    if result.stdout.strip():
        raise RuntimeError(
            "GPU is occupied; evaluation will not interrupt existing jobs: "
            + result.stdout.strip()
        )


def safe_environment(root):
    allowed = (
        "PATH",
        "LANG",
        "LC_ALL",
        "CUDA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "HF_HOME",
        "HF_HUB_CACHE",
        "UV_CACHE_DIR",
        "TMPDIR",
    )
    env = {k: os.environ[k] for k in allowed if k in os.environ}
    env.update(
        HOME=str(root / "empty-home"),
        PYTHONPATH=str(REPO),
        PYTHONNOUSERSITE="1",
        TOKENIZERS_PARALLELISM="false",
        WANDB_DISABLED="true",
        HARBOR_TELEMETRY="false",
        APPTAINER_CLEANENV="1",
        SINGULARITY_CLEANENV="1",
        APPTAINER_UNSQUASH="1",
        SINGULARITY_UNSQUASH="1",
        APPTAINER_WRITABLE="1",
        SINGULARITY_WRITABLE="1",
    )
    Path(env["HOME"]).mkdir(parents=True, exist_ok=True)
    env["PATH"] = str(root / "eval-bin") + ":" + env.get("PATH", "/usr/bin:/bin")
    return env


def runtime_mounts(directory):
    """Disposable per-container plumbing only; never bind host configuration itself."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = {
        "resolv.conf": ("/etc/resolv.conf", Path("/etc/resolv.conf").read_text()),
        "apt.conf": (
            "/etc/apt/apt.conf.d/99harbor-rootless",
            'APT::Sandbox::User "root";\nAcquire::Retries "0";\nAcquire::http::Timeout "20";\n',
        ),
    }
    mounts = []
    for name, (target, content) in files.items():
        source = directory / name
        source.write_text(content)
        source.chmod(0o644)
        mounts.append(
            dict(type="bind", source=str(source), target=target, read_only=True)
        )
    write(
        directory / "manifest.json",
        dict(
            unsquash=True,
            writable_temporary_sandbox=True,
            files={name: sha(directory / name) for name in files},
            note="Disposable copies only; pinned Harbor backend does not honor per-bind read_only flags.",
        ),
    )
    return mounts


def runtime_image_definition(source):
    # Writable extraction disables automatic overlay creation of bind targets.
    # Add empty harness mountpoints only; leave the task payload untouched.
    return (
        f"Bootstrap: localimage\nFrom: {source}\n\n%post\n"
        "    mkdir -p /staging/env_files /logs/agent /logs/verifier /logs/artifacts /etc/apt/apt.conf.d\n"
        "    touch /etc/apt/apt.conf.d/99harbor-rootless\n"
    )


def image_digest(image):
    # The selected official images currently live on Docker Hub. Fail on unhandled registries.
    image = image.removeprefix("docker://")
    if "@sha256:" in image:
        return image
    repository, tag = image.rsplit(":", 1) if ":" in image else (image, "latest")
    if repository.startswith("docker.io/"):
        repository = repository[len("docker.io/") :]
    if "." in repository.split("/")[0] or ":" in repository.split("/")[0]:
        raise ValueError(f"Unrecognized image registry: {image}")
    if "/" not in repository:
        repository = "library/" + repository
    url = "https://auth.docker.io/token?" + urllib.parse.urlencode(
        dict(service="registry.docker.io", scope=f"repository:{repository}:pull")
    )
    with urllib.request.urlopen(url, timeout=30) as response:
        token = json.load(response)["token"]
    req = urllib.request.Request(
        f"https://registry-1.docker.io/v2/{repository}/manifests/{tag}",
        headers={
            "Authorization": "Bearer " + token,
            "Accept": ", ".join(
                [
                    "application/vnd.oci.image.index.v1+json",
                    "application/vnd.docker.distribution.manifest.list.v2+json",
                    "application/vnd.oci.image.manifest.v1+json",
                    "application/vnd.docker.distribution.manifest.v2+json",
                ]
            ),
        },
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        data = response.read()
        digest = "sha256:" + hashlib.sha256(data).hexdigest()
        if response.headers.get("Docker-Content-Digest", digest) != digest:
            raise ValueError("OCI manifest digest mismatch")
        manifest = json.loads(data)
    if "manifests" in manifest:
        choices = [
            m
            for m in manifest["manifests"]
            if m.get("platform", {}).get("os") == "linux"
            and m["platform"].get("architecture") == "amd64"
        ]
        if len(choices) != 1:
            raise ValueError("Ambiguous Linux amd64 image")
        digest = choices[0]["digest"]
    return repository + "@" + digest


def trial_order(tasks):
    return [
        dict(task=task, model=model, id=f"{i:02d}-{task}-{model}")
        for i, task in enumerate(tasks)
        for model in (("base", "sft") if i % 2 == 0 else ("sft", "base"))
    ]


def reward_from_result(result):
    reward = ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    if (
        isinstance(reward, bool)
        or not isinstance(reward, (int, float))
        or not math.isfinite(reward)
    ):
        return None
    return reward


def seconds(timing):
    if not timing or not timing.get("started_at") or not timing.get("finished_at"):
        return None
    return (
        datetime.fromisoformat(timing["finished_at"])
        - datetime.fromisoformat(timing["started_at"])
    ).total_seconds()


class Evaluation:
    def __init__(self, args):
        self.args = args
        self.root, self.directory = args.root.resolve(), args.run_dir.resolve()
        self.state_path = self.directory / "state.json"
        self.processes = []
        self.stopping = False
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory / ".lock").open("w")
        if not args.worker:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.env = safe_environment(self.root)
        self.env["TB_EVALUATION_RUN"] = str(self.directory)
        self.reporter = None
        self.session_id = uuid.uuid4().hex
        self.state = read(self.state_path)
        if self.state:
            if not args.resume and args.phase != "report":
                raise ValueError(
                    "Run exists; pass --resume to preserve its deadline and attempts"
                )
            if (
                args.sft_run.resolve() != Path(self.state["sft_run"])
                or args.tasks != self.state["tasks"]
            ):
                raise ValueError("Resume configuration does not match saved run")
            if not args.worker:
                self.cleanup_abandoned_groups()
            for trial in [
                *self.state["trials"].values(),
                *self.state["controls"].values(),
            ]:
                if trial["status"] == "started":
                    trial["status"] = "interrupted"
                    trial["error"] = (
                        "Supervisor interrupted; attempt retained and never repeated"
                    )
        else:
            now = time.time()
            self.state = dict(
                started_at=now,
                deadline=now + args.hours * 3600,
                sft_run=str(args.sft_run.resolve()),
                tasks=args.tasks,
                order=trial_order(args.tasks),
                trials={},
                controls={},
                phases={},
                status="created",
            )
        self.save()

    def save(self):
        write(self.state_path, self.state)

    @staticmethod
    def process_identity(pid):
        try:
            stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            env = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
            return dict(start_ticks=stat[19], environment=env, pgid=os.getpgid(pid))
        except (OSError, IndexError):
            return None

    def cleanup_abandoned_groups(self):
        journal = self.directory / "processes.jsonl"
        if not journal.exists():
            return
        for line in journal.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not record.get("owns_group"):
                continue
            identity = self.process_identity(record["pid"])
            if identity is None:
                continue
            owned = (
                identity["start_ticks"] == record.get("start_ticks")
                and identity["pgid"] == record["pid"]
                and f"TB_EVALUATION_RUN={self.directory}".encode()
                in identity["environment"]
            )
            if owned:
                os.killpg(record["pid"], signal.SIGTERM)
                until = time.monotonic() + 2
                while self.process_identity(record["pid"]) and time.monotonic() < until:
                    time.sleep(0.05)
                try:
                    os.killpg(record["pid"], signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def event(self, phase, **values):
        elapsed = time.time() - self.state["started_at"]
        done = sum(t["status"] != "started" for t in self.state["trials"].values())
        if not hasattr(self, "event_args"):
            self.event_args = SimpleNamespace(
                event_file=self.directory / "events" / f"{self.session_id}.jsonl",
                session_id=self.session_id,
                phase=phase,
            )
        emit(
            self.event_args,
            phase,
            completed_trials=done,
            elapsed_seconds=elapsed,
            remaining_seconds=max(0, self.state["deadline"] - time.time()),
            **values,
        )
        print(
            f"[{phase}] {done}/10 trials elapsed={elapsed:.0f}s remaining={max(0, self.state['deadline'] - time.time()):.0f}s {values}",
            flush=True,
        )
        write(
            self.directory / "progress.json",
            dict(
                phase=phase,
                completed_trials=done,
                planned_trials=10,
                elapsed_seconds=elapsed,
                remaining_seconds=max(0, self.state["deadline"] - time.time()),
                **values,
            ),
        )

    def remaining(self):
        return self.state["deadline"] - 60 - time.time()

    def check_deadline(self):
        if self.stopping or self.remaining() <= 0:
            raise TimeoutError(
                "Evaluation deadline reached; reserving 60 seconds for cleanup/reporting"
            )

    def spawn(self, command, label, env=None):
        self.check_deadline()
        log = (self.directory / f"{label}.log").open("ab", buffering=0)
        p = subprocess.Popen(
            [str(x) for x in command],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=self.env if env is None else env,
            start_new_session=not self.args.worker,
            cwd=self.directory,
        )
        p.owns_group = not self.args.worker
        log.close()
        self.processes.append(p)
        # Record ownership for diagnostics; never kill an unverified PID on resume.
        with (self.directory / "processes.jsonl").open("a") as stream:
            identity = self.process_identity(p.pid)
            stream.write(
                json.dumps(
                    dict(
                        pid=p.pid,
                        label=label,
                        at=time.time(),
                        owns_group=p.owns_group,
                        start_ticks=identity["start_ticks"] if identity else None,
                    )
                )
                + "\n"
            )
        return p

    @staticmethod
    def terminate(p, grace=8):
        if p.poll() is None:
            try:
                if getattr(p, "owns_group", True):
                    os.killpg(p.pid, signal.SIGTERM)
                else:
                    p.terminate()
                p.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                if getattr(p, "owns_group", True):
                    os.killpg(p.pid, signal.SIGKILL)
                else:
                    p.kill()
                p.wait(timeout=3)
            except ProcessLookupError:
                pass
        # A worker may have exited while descendants remain in its private process group.
        if getattr(p, "owns_group", True):
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def command(self, command, label):
        p = self.spawn(command, label)
        next_update = 0
        try:
            while p.poll() is None:
                self.check_deadline()
                self.poll_reporter()
                if time.time() >= next_update:
                    if label == "evaluation-worker":
                        progress = read(self.directory / "progress.json", {})
                        print(f"[supervisor] {json.dumps(progress)}", flush=True)
                    else:
                        self.event(label, status="running")
                    next_update = time.time() + 30
                time.sleep(0.2)
            if p.returncode:
                # Surface the failing operation instead of hiding it behind two
                # layers of supervisor errors. Full logs stay local.
                log_path = self.directory / f"{label}.log"
                with log_path.open("rb") as stream:
                    stream.seek(0, os.SEEK_END)
                    stream.seek(max(0, stream.tell() - 6000))
                    tail = stream.read().decode(errors="replace").strip()
                raise RuntimeError(
                    f"{label} failed with exit {p.returncode}; see {log_path}\n{tail}"
                )
        finally:
            self.terminate(p)

    def start_reporter(self):
        if self.args.wandb_mode == "disabled":
            return
        py = self.root / "tracking-venv/bin/python"
        ready = read(self.root / "tracking-venv/ready.json", {})
        if not py.exists() or ready.get("lock_sha256") != sha(
            REPO / "requirements-tracking.lock"
        ):
            raise ValueError(
                "Tracking environment missing/outdated; run ./setup.sh --setup-tracking"
            )
        env = dict(self.env)
        env.pop("WANDB_DISABLED", None)
        if "WANDB_API_KEY" in os.environ:
            env["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
        tracking = self.directory / "tracking"
        tracking.mkdir(exist_ok=True)
        (tracking / "DONE").unlink(missing_ok=True)
        self.reporter_started = time.time()
        self.reporter = self.spawn(
            [
                py,
                REPO / "terminalbench_tracking.py",
                self.directory,
                self.args.wandb_mode,
            ],
            "reporter",
            env=env,
        )

    def poll_reporter(self):
        if self.reporter is None:
            return
        status = read(self.directory / "tracking/status.json", {})
        if (
            self.reporter.poll() is not None
            or time.time() - status.get("heartbeat", self.reporter_started) > 45
        ):
            self.terminate(self.reporter, grace=2)
            write(
                self.directory / "tracking/controller.json",
                dict(state="unavailable", local_journal_retained=True),
            )
            self.reporter = None

    def preflight(self):
        self.event("preflight", status="checking provenance and selected images")
        for name in ("harbor", "vllm"):
            py = self.root / f"eval-{name}-venv/bin/python"
            ready = read(self.root / f"eval-{name}-venv/ready.json")
            if (
                not py.exists()
                or not ready
                or ready["lock_sha256"] != sha(REPO / f"requirements-eval-{name}.lock")
            ):
                raise ValueError(
                    "Missing/stale eval environment; run ./setup.sh --setup-eval"
                )
        checkpoint, experiment = verify_checkpoint(Path(self.state["sft_run"]))
        if (
            sha(self.root / "requirements.lock")
            != experiment["identity"]["dependency_lock_sha256"]
        ):
            raise ValueError(
                "Training dependency lock changed since the approved SFT run"
            )
        source_revision = subprocess.check_output(
            ["git", "-C", str(self.root / "LLaMA-Factory"), "rev-parse", "HEAD"],
            text=True,
            timeout=15,
        ).strip()
        if source_revision != "8ebba0d07dd14fdb1d429c497f018c7dae2e44b7":
            raise ValueError("Training source revision mismatch")
        subprocess.run(
            [
                "git",
                "-C",
                str(self.root / "LLaMA-Factory"),
                "diff",
                "--quiet",
                "HEAD",
                "--",
                "src",
            ],
            check=True,
            timeout=15,
        )
        self.check_deadline()
        # Verify the source installation, not just its advertised package version.
        self.command(
            [
                self.root / "eval-harbor-venv/bin/python",
                "-c",
                "import importlib.metadata as m,json; u=json.loads(m.distribution('harbor').read_text('direct_url.json')); "
                f"assert u['vcs_info']['commit_id']=='{HARBOR}'",
            ],
            "harbor-revision",
        )
        from huggingface_hub import snapshot_download

        source = Path(
            snapshot_download(
                DATASET,
                repo_type="dataset",
                revision=REVISION,
                allow_patterns=[f"tasks/{t}/**" for t in self.args.tasks],
            )
        )
        if source.name != REVISION:
            raise ValueError("Dataset revision mismatch")
        subset = self.directory / "tasks"
        manifest = dict(
            dataset=DATASET,
            version="2.1.0",
            revision=REVISION,
            tasks={},
            resource_note=RESOURCE_NOTE,
        )
        instructions = {}
        for name in self.args.tasks:
            self.check_deadline()
            original = source / "tasks" / name
            config = tomllib.loads((original / "task.toml").read_text())
            for phase in ("agent", "verifier"):
                if config[phase]["timeout_sec"] != 900:
                    raise ValueError(f"Unexpected official {phase} timeout for {name}")
            if config.get("environment", {}).get("gpus", 0) != 0:
                raise ValueError("Task requests GPU access")
            digest = image_digest(config["environment"]["docker_image"])
            sif = self.directory / "images" / (digest.split("@sha256:")[1] + ".sif")
            sif.parent.mkdir(exist_ok=True)
            if not sif.exists():
                self.command(
                    ["singularity", "pull", str(sif), "docker://" + digest],
                    f"pull-{name}",
                )
            source_sif = sif
            source_sha256 = sha(source_sif)
            definition = source_sif.with_suffix(".def")
            definition.write_text(runtime_image_definition(source_sif))
            sif = source_sif.with_name(source_sif.stem + "-harbor.sif")
            if not sif.exists():
                self.command(
                    [
                        "singularity",
                        "build",
                        "--fakeroot",
                        "--notest",
                        str(sif),
                        str(definition),
                    ],
                    f"runtime-image-{name}",
                )
            if sha(source_sif) != source_sha256:
                raise ValueError(
                    "Original task image changed while deriving harness mountpoints"
                )
            # Probe the selected image and mount isolation with the same containment flags.
            self.command(
                [
                    "singularity",
                    "exec",
                    "--containall",
                    "--fakeroot",
                    "--writable-tmpfs",
                    "--no-mount",
                    "home,tmp,bind-paths,cwd,hostfs",
                    str(sif),
                    "sh",
                    "-c",
                    'command -v bash && test ! -e "$1" && test ! -e "$2"',
                    "probe",
                    str(REPO),
                    str(Path(self.state["sft_run"])),
                ],
                f"probe-{name}",
            )
            target = subset / name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(original, target)
            text = (target / "task.toml").read_text()
            text = text.replace(
                "docker_image = " + json.dumps(config["environment"]["docker_image"]),
                "docker_image = " + json.dumps(str(sif)),
            )
            (target / "task.toml").write_text(text)
            instructions[name] = (original / "instruction.md").read_text()
            manifest["tasks"][name] = dict(
                original_files=hashes(original),
                effective_files=hashes(target),
                declared=config,
                resolved_image=digest,
                source_sif=str(source_sif),
                source_sif_sha256=source_sha256,
                runtime_definition=str(definition),
                runtime_definition_sha256=sha(definition),
                sif_sha256=sha(sif),
                sif=str(sif),
            )
        write(self.directory / "subset.json", manifest)
        overlap = overlaps(Path(self.state["sft_run"]), instructions)
        write(self.directory / "overlaps.json", overlap)
        if overlap["exact"]:
            raise ValueError(
                "Exact normalized training/validation instruction overlap; see overlaps.json"
            )
        provenance = dict(
            harbor_commit=HARBOR,
            vllm="0.34.0",
            dataset_revision=REVISION,
            model_revision=experiment["identity"]["revisions"]["model_revision"],
            checkpoint=str(checkpoint),
            checkpoint_files=hashes(checkpoint),
            experiment_sha256=sha(Path(self.state["sft_run"]) / "experiment.json"),
            description_sources={
                name: sha(Path(self.state["sft_run"]) / name)
                for name in (
                    "selected_trajectories.json",
                    "context-32768/data/manifest.json",
                )
            },
            source_hashes={p.name: sha(p) for p in REPO.glob("terminalbench*.py")},
            locks={
                n: sha(REPO / f"requirements-eval-{n}.lock") for n in ("harbor", "vllm")
            },
            training_lock=sha(self.root / "requirements.lock"),
            context=CONTEXT,
            output=OUTPUT,
            seed=42,
            gpu_memory_utilization=0.40,
            max_num_seqs=1,
            numerical_validation_policy="advisory",
            numerical_validation_note="User requested evaluation with numerical differences documented for future investigation.",
            thinking=TEMPLATE_KWARGS,
            container_runtime=dict(
                unsquash=True,
                writable_temporary_sandbox=True,
                added_mount_targets=[
                    "/etc/resolv.conf",
                    "/etc/apt/apt.conf.d/99harbor-rootless",
                ],
                disposable_per_trial_files=True,
                backend_honors_read_only_bind_flags=False,
            ),
            sft_window=32768,
            extends_sft_window=True,
            resource_note=RESOURCE_NOTE,
        )
        write(self.directory / "provenance.json", provenance)
        self.state["phases"]["preflight"] = True
        self.state["status"] = "preflight_complete"
        self.save()

    def verify_inputs(self):
        provenance = read(self.directory / "provenance.json")
        if not provenance or not self.state["phases"].get("preflight"):
            raise ValueError("Run preflight first")
        if provenance["source_hashes"] != {
            p.name: sha(p) for p in REPO.glob("terminalbench*.py")
        }:
            raise ValueError("Evaluation source changed; use a new run directory")
        for name, value in provenance["locks"].items():
            if sha(REPO / f"requirements-eval-{name}.lock") != value:
                raise ValueError("Evaluation dependency lock changed")
        if sha(self.root / "requirements.lock") != provenance["training_lock"]:
            raise ValueError("Training reference environment lock changed")
        for name, digest in provenance["description_sources"].items():
            if sha(Path(self.state["sft_run"]) / name) != digest:
                raise ValueError("Saved training/validation descriptions changed")
        checkpoint, experiment = verify_checkpoint(Path(self.state["sft_run"]))
        if hashes(checkpoint) != provenance["checkpoint_files"]:
            raise ValueError("Original checkpoint files changed")
        if (
            sha(Path(self.state["sft_run"]) / "experiment.json")
            != provenance["experiment_sha256"]
        ):
            raise ValueError("Original experiment changed")
        subset = read(self.directory / "subset.json")
        for name, record in subset["tasks"].items():
            if (
                sha(record["source_sif"]) != record["source_sif_sha256"]
                or sha(record["runtime_definition"])
                != record["runtime_definition_sha256"]
            ):
                raise ValueError(
                    f"Original image/runtime definition integrity changed: {name}"
                )
            if (
                hashes(self.directory / "tasks" / name) != record["effective_files"]
                or sha(record["sif"]) != record["sif_sha256"]
            ):
                raise ValueError(f"Task/image integrity changed: {name}")

    def prepare(self):
        self.verify_inputs()
        gpu_free()
        self.event(
            "prepare",
            status="merging BF16 evaluation copy and checking Transformers equivalence",
        )
        self.command(
            [
                self.root / "venv/bin/python",
                REPO / "terminalbench_prepare.py",
                self.directory,
            ],
            "prepare",
        )
        self.state["phases"]["prepare"] = True
        self.state["status"] = "prepared"
        self.save()

    def start_servers(self):
        gpu_free()
        prepared = read(self.directory / "prepared.json")
        if not self.state["phases"].get("prepare") or not prepared:
            raise ValueError("Run prepare first")
        if hashes(prepared["sft"]) != prepared["merged_hashes"]:
            raise ValueError("Merged evaluation copy integrity failure")
        if (
            hashes(prepared["base"], names=prepared["base_hashes"])
            != prepared["base_hashes"]
        ):
            raise ValueError("Base model cache integrity failure")
        if sha(self.directory / "reference.json") != prepared["reference_sha256"]:
            raise ValueError("Reference values changed")
        loading = prepared.get("adapter_loading", {})
        if (
            loading.get("exact_match") is not True
            or loading.get("tensor_count", 0) <= 0
        ):
            raise ValueError(
                "Adapter loading was not verified; prepare a new evaluation copy"
            )
        self.endpoints = {}
        self.servers = {}
        for model in ("base", "sft"):
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            base = f"http://127.0.0.1:{port}"
            command = [
                self.root / "eval-vllm-venv/bin/python",
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                prepared[model],
                "--served-model-name",
                model,
                "--tokenizer",
                prepared["tokenizer"],
                "--chat-template",
                str(Path(prepared["tokenizer"]) / "chat_template.jinja"),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--dtype",
                "bfloat16",
                "--max-model-len",
                str(CONTEXT),
                "--gpu-memory-utilization",
                "0.40",
                "--max-num-seqs",
                "1",
                "--seed",
                "42",
                "--reasoning-parser",
                "qwen3",
                "--generation-config",
                "vllm",
            ]
            p = self.spawn(command, f"server-{model}")
            self.servers[model] = p
            self.event("servers", model=model, status="starting sequentially")
            next_update = time.time() + 30
            startup_deadline = time.time() + 600
            while True:
                self.check_deadline()
                if p.poll() is not None or time.time() > startup_deadline:
                    raise RuntimeError(
                        f"Both BF16 servers must fit without changing settings; inspect server-{model}.log"
                    )
                try:
                    models = request(base, "/v1/models", timeout=2)
                    if models["data"][0]["id"] != model:
                        raise ValueError("Model routing mismatch")
                    break
                except (OSError, KeyError):
                    time.sleep(0.5)
                if time.time() >= next_update:
                    self.event("servers", model=model, status="loading")
                    next_update = time.time() + 30
            self.endpoints[model] = base
            self.serving_parity(model, base)
        self.check_servers()
        self.event(
            "servers", status="both models ready; numerical diagnostics recorded"
        )

    def check_servers(self):
        for model, process in self.servers.items():
            if process.poll() is not None:
                raise RuntimeError(
                    f"Evaluation server {model} exited; inspect server-{model}.log"
                )
            if (
                request(self.endpoints[model], "/v1/models", timeout=3)["data"][0]["id"]
                != model
            ):
                raise ValueError("Evaluation server routing changed")

    def serving_parity(self, model, base):
        references = read(self.directory / "reference.json")
        differences = []
        renderings = []
        chosen_values = []
        for probe, expected in zip(references["probes"], references[model]):
            self.check_deadline()
            from terminalbench_qwen import token_payload

            tokens = request(base, "/tokenize", token_payload(model, probe["messages"]))
            renderings.append(dict(expected=probe["prompt"], observed=tokens["tokens"]))
            write(self.directory / f"serving-rendering-{model}.json", renderings)
            if tokens["tokens"] != probe["prompt"]:
                raise ValueError(
                    f"vLLM and Transformers rendered template/token IDs differ; "
                    f"inspect serving-rendering-{model}.json"
                )
            response = request(
                base,
                "/v1/completions",
                dict(
                    model=model,
                    prompt=probe["prompt"] + probe["target"],
                    max_tokens=1,
                    temperature=0,
                    top_p=1,
                    seed=42,
                    logprobs=1,
                    echo=True,
                ),
                timeout=120,
            )
            actual = response["choices"][0]["logprobs"]["token_logprobs"]
            actual = actual[len(probe["prompt"]) : len(probe["prompt"]) + len(expected)]
            if len(actual) != len(expected) or any(v is None for v in actual):
                raise ValueError("vLLM did not return chosen prompt-token logprobs")
            chosen_values.append(
                dict(target_ids=probe["target"], transformers=expected, vllm=actual)
            )
            write(self.directory / f"serving-values-{model}.json", chosen_values)
            differences.extend(abs(a - b) for a, b in zip(actual, expected))
        parity = dict(
            mean_chosen_token_difference=sum(differences) / len(differences),
            maximum_difference=max(differences),
            tolerance=0.02,
            policy="advisory",
        )
        parity["within_tolerance"] = parity["mean_chosen_token_difference"] <= 0.02
        write(self.directory / f"serving-parity-{model}.json", parity)
        if not parity["within_tolerance"]:
            self.event("numerical-advisory", model=model, **parity)
        # Exercise the actual chat route too: its history normalization differs
        # from /tokenize in this vLLM release. This is a short non-benchmark probe.
        payload = token_payload(model, references["probes"][-1]["messages"])
        exact_count = request(base, "/tokenize", payload)["count"]
        chat = request(
            base,
            "/v1/chat/completions",
            dict(
                **payload,
                max_tokens=128,
                temperature=0,
                top_p=1,
                seed=42,
                reasoning_effort="xhigh",
            ),
            timeout=120,
        )
        write(
            self.directory / f"serving-chat-{model}.json",
            dict(
                exact_prompt_tokens=exact_count,
                usage=chat["usage"],
                finish_reason=chat["choices"][0]["finish_reason"],
            ),
        )
        if chat["usage"]["prompt_tokens"] != exact_count:
            raise ValueError(
                "vLLM chat/tokenizer prompt counts disagree on preserved reasoning"
            )

    def trial_config(self, task, model, trial_id):
        mounts = runtime_mounts(self.directory / "runtime" / trial_id)
        # Explicit output mounts avoid the pinned backend's staging download
        # path and let Harbor read the verifier's reward directly from its log dir.
        for relative, target in (
            ("agent", "/logs/agent"),
            ("verifier", "/logs/verifier"),
            ("artifacts/logs/artifacts", "/logs/artifacts"),
        ):
            source = self.directory / "harbor" / trial_id / relative
            source.mkdir(parents=True, exist_ok=True)
            mounts.append(dict(type="bind", source=str(source), target=target))
        agent = (
            dict(name=model)
            if model in ("oracle", "nop")
            else dict(
                import_path="terminalbench_qwen:QwenTerminus2",
                model_name=f"openai/{model}",
                base_url=self.endpoints[model] + "/v1",
                model_api_format="openai_chat_completions",
                skip_model_parsing=True,
                kwargs=dict(
                    temperature=0,
                    reasoning_effort="xhigh",
                    interleaved_thinking=True,
                    enable_summarize=True,
                    proactive_summarization_threshold=SUMMARIZE,
                    parser_name="json",
                    model_info=dict(max_input_tokens=CONTEXT, max_output_tokens=OUTPUT),
                ),
            )
        )
        return dict(
            task=dict(path=str(self.directory / "tasks" / task)),
            trial_name=trial_id,
            trials_dir=str(self.directory / "harbor"),
            agent=agent,
            environment=dict(
                type="singularity",
                delete=True,
                cpu_enforcement_policy="auto",
                memory_enforcement_policy="auto",
                mounts=mounts,
                kwargs=dict(
                    singularity_no_mount="home,tmp,bind-paths,cwd,hostfs",
                    singularity_image_cache_dir=str(self.directory / "images"),
                ),
            ),
        )

    def trial(self, task, model, trial_id, control=False):
        self.check_deadline()
        bucket = self.state["controls" if control else "trials"]
        if trial_id in bucket:
            return bucket[trial_id]
        config = self.trial_config(task, model, trial_id)
        config_path = self.directory / "configs" / f"{trial_id}.json"
        write(config_path, config)
        record = dict(
            task=task,
            model=model,
            id=trial_id,
            status="started",
            started_at=time.time(),
        )
        bucket[trial_id] = record
        self.save()  # Publish started BEFORE spawning; resume never repeats an uncertain attempt.
        self.event("trial", task=task, model=model, status="started")
        env = dict(self.env)
        if not control:
            env.update(TB_MODEL=model, TB_ENDPOINT=self.endpoints[model])
        p = self.spawn(
            [
                self.root / "eval-harbor-venv/bin/python",
                REPO / "terminalbench_worker.py",
                config_path,
            ],
            trial_id,
            env=env,
        )
        next_update = time.time() + 30
        try:
            while p.poll() is None:
                self.check_deadline()
                if time.time() >= next_update:
                    self.event("trial", task=task, model=model, status="running")
                    next_update = time.time() + 30
                time.sleep(0.2)
            record.update(
                status="finished" if p.returncode == 0 else "infrastructure_failure",
                exit_code=p.returncode,
            )
        except (TimeoutError, KeyboardInterrupt):
            record.update(
                status="interrupted",
                error="Deadline or shutdown interrupted agent/verifier work",
            )
            raise
        finally:
            self.terminate(p, grace=15)
            record["finished_at"] = time.time()
            self.collect_result(record)
            self.save()
            self.report()
        if record["status"] == "infrastructure_failure":
            raise RuntimeError(
                f"Harbor worker crashed for {trial_id}; stopping for container cleanup"
            )
        return record

    def collect_result(self, record):
        directory = self.directory / "harbor" / record["id"]
        result_path = directory / "result.json"
        try:
            result = read(result_path, {})
        except (json.JSONDecodeError, OSError) as exc:
            result = {}
            record["infrastructure_failure"] = (
                f"Unreadable Harbor result: {type(exc).__name__}"
            )
        record.update(
            reward=reward_from_result(result),
            result_path=str(result_path.relative_to(self.directory)),
            exception_info=result.get("exception_info"),
            reward_valid=result.get("reward_valid"),
            agent_seconds=seconds(result.get("agent_execution")),
            verifier_seconds=seconds(result.get("verifier")),
        )
        if not result:
            record.setdefault(
                "infrastructure_failure", "Missing Harbor result; no verified reward"
            )
        if result.get("exception_info") and result.get("reward_valid") is False:
            record["infrastructure_failure"] = result["exception_info"][
                "exception_type"
            ]
        record["unfinished_agent"] = (
            bool(
                result.get("agent_execution", {}).get("started_at")
                and not result.get("agent_execution", {}).get("finished_at")
            )
            if result.get("agent_execution")
            else record["status"] == "interrupted"
        )
        record["unfinished_verifier"] = (
            bool(
                result.get("verifier", {}).get("started_at")
                and not result.get("verifier", {}).get("finished_at")
            )
            if result.get("verifier")
            else record["status"] == "interrupted"
        )
        record["trajectories"] = [
            str(p.relative_to(self.directory))
            for p in directory.rglob("trajectory*.json")
        ]
        calls = []
        telemetry = []
        for path in directory.rglob("inference.jsonl"):
            for line in path.read_text().splitlines():
                try:
                    telemetry.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        calls = [r for r in telemetry if r.get("kind") == "inference"]
        record.update(
            prompt_tokens=sum(x["prompt_tokens"] for x in calls),
            completion_tokens=sum(x["completion_tokens"] for x in calls),
            inference_seconds=sum(x["inference_seconds"] for x in calls),
            truncated_completions=sum(x["truncated"] for x in calls),
            reasoning_truncations=sum(x["reasoning_truncated"] for x in calls),
            llm_calls=len(calls),
        )
        record["agent_turns"] = sum(r.get("kind") == "turn" for r in telemetry)
        record["json_parse_failures"] = sum(
            bool(r.get("json_parse_failure")) for r in telemetry
        )
        durations = [
            r["terminal_seconds"] for r in telemetry if r.get("kind") == "terminal"
        ]
        record["terminal_seconds"] = sum(durations) if durations else None

    def calibrate(self):
        for task in self.args.tasks:
            for model, expected in (("oracle", 1), ("nop", 0)):
                trial_id = f"control-{task}-{model}"
                record = self.trial(task, model, trial_id, control=True)
                if (
                    record.get("reward") != expected
                    or record.get("exception_info")
                    or record["status"] != "finished"
                ):
                    error = record.get("exception_info") or {}
                    raise RuntimeError(
                        f"Calibration failure: {task}/{model}, expected reward={expected}, "
                        f"recorded reward={record.get('reward')}; "
                        f"{error.get('exception_type', '')}: {error.get('exception_message', '')}. "
                        f"Inspect {self.directory / 'harbor' / trial_id}"
                    )
        self.state["phases"]["calibrate"] = True
        self.state["status"] = "calibrated"
        self.save()

    def run(self):
        self.verify_inputs()
        self.calibrate()
        if self.args.phase == "calibrate":
            return
        self.start_servers()
        for entry in self.state["order"]:
            self.check_servers()
            self.trial(entry["task"], entry["model"], entry["id"])
        self.state["phases"]["run"] = True
        self.state["status"] = (
            "complete"
            if all(t.get("reward") is not None for t in self.state["trials"].values())
            else "partial"
        )
        self.save()

    def report(self):
        for record in self.state["trials"].values():
            self.collect_result(record)
        trials = list(self.state["trials"].values())
        scored = [t for t in trials if t.get("reward") is not None]
        models = {
            model: dict(
                solved=sum(t["reward"] == 1 for t in scored if t["model"] == model),
                scored=sum(t["model"] == model for t in scored),
                planned=5,
            )
            for model in ("base", "sft")
        }
        paired = dict(base_wins=0, sft_wins=0, ties=0, tasks=[])
        for task in self.args.tasks:
            rewards = {t["model"]: t["reward"] for t in scored if t["task"] == task}
            if len(rewards) == 2:
                paired["tasks"].append(task)
                key = (
                    "ties"
                    if rewards["base"] == rewards["sft"]
                    else (
                        "base_wins" if rewards["base"] > rewards["sft"] else "sft_wins"
                    )
                )
                paired[key] += 1
        summary = dict(
            status=self.state["status"],
            models=models,
            paired=paired,
            scored_trials=len(scored),
            integration_check=self.args.tasks != TASKS,
            configured_trials=len(self.state["order"]),
            planned_trials=10,
            attempted_trials=len(trials),
            infrastructure_failures=sum(
                bool(t.get("infrastructure_failure")) for t in trials
            ),
            calibration_failures=sum(
                t.get("status") != "finished"
                or t.get("exception_info") is not None
                or t.get("reward") != (1 if t["model"] == "oracle" else 0)
                for t in self.state["controls"].values()
            ),
            unstarted_trials=10 - len(trials),
            elapsed_seconds=time.time() - self.state["started_at"],
            deadline=self.state["deadline"],
            controls=self.state["controls"],
            trials=trials,
            resource_note=RESOURCE_NOTE,
            numerical_checks=dict(
                policy="advisory",
                merge=read(self.directory / "merge-equivalence.json"),
                serving={
                    model: read(self.directory / f"serving-parity-{model}.json")
                    for model in ("base", "sft")
                },
            ),
            limitations=[
                "Exploratory five-task subset; no leaderboard or significance claims.",
                "64K evaluation extends beyond the adapter's 32K SFT window.",
                "Missing rewards remain unscored. Recorded verifier reward is independent of agent completion claims.",
                "Instruction overlap checks cannot exclude broader contamination.",
                "Numerical equivalence checks are advisory by user instruction. BF16 merging and vLLM/Transformers probability differences remain future investigation; see merge-equivalence.json and serving-parity-*.json.",
            ],
            error=self.state.get("error"),
        )
        write(self.directory / "summary.json", summary)
        columns = [
            "id",
            "task",
            "model",
            "status",
            "reward",
            "reward_valid",
            "unfinished_agent",
            "unfinished_verifier",
            "agent_turns",
            "json_parse_failures",
            "prompt_tokens",
            "completion_tokens",
            "truncated_completions",
            "reasoning_truncations",
            "inference_seconds",
            "agent_seconds",
            "terminal_seconds",
            "verifier_seconds",
            "infrastructure_failure",
            "error",
            "exception_info",
            "result_path",
        ]
        with (self.directory / "trials.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(trials)
        lines = [
            "# Terminal-Bench 2.1 pilot",
            "",
            f"Status: {summary['status']}. Scored coverage: {len(scored)}/10 planned trials.",
            "",
            f"Base solved {models['base']['solved']}/5 planned tasks ({models['base']['scored']} scored); "
            f"SFT checkpoint 190 solved {models['sft']['solved']}/5 ({models['sft']['scored']} scored).",
            "",
            f"Paired tasks: {len(paired['tasks'])}; base wins {paired['base_wins']}, SFT wins {paired['sft_wins']}, ties {paired['ties']}.",
            "",
            "| Task | Model | Status | Official reward | Agent/verifier error | Trajectories |",
            "|---|---|---|---|---|---|",
        ]
        for t in trials:
            links = " ".join(f"[ATIF]({p})" for p in t.get("trajectories", []))
            lines.append(
                f"| {t['task']} | {t['model']} | {t['status']} | {t.get('reward') if t.get('reward') is not None else 'unscored'} | "
                f"{(t.get('exception_info') or {}).get('exception_type') or t.get('error') or ''} | {links} |"
            )
        lines.extend(
            [
                "",
                RESOURCE_NOTE,
                "",
                *summary["limitations"],
                "",
                "See [per-trial metrics](trials.csv), [summary](summary.json), [provenance](provenance.json), "
                "[subset](subset.json), and [overlap audit](overlaps.json). Agent errors and partial verifier work remain in local Harbor logs.",
            ]
        )
        if self.state.get("error"):
            lines.extend(["", "Run error: " + self.state["error"]])
        (self.directory / "report.md").write_text("\n".join(lines) + "\n")
        self.save()

    def close(self):
        self.report()
        if self.reporter is not None:
            (self.directory / "tracking/DONE").touch()
            try:
                self.reporter.wait(timeout=6)
            except subprocess.TimeoutExpired:
                self.terminate(self.reporter, grace=2)
        for p in reversed(self.processes):
            self.terminate(p)
        self.report()
        self.event("final", status=self.state["status"])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        nargs="?",
        choices=["all", "preflight", "prepare", "calibrate", "run", "report"],
        default="all",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            os.environ.get(
                "SMOKE_ROOT",
                "/scratch1/"
                + os.environ.get("USER", "mendeza")
                + "/qwen38-27b-llamafactory-test",
            )
        ),
    )
    parser.add_argument("--sft-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--wandb-mode",
        choices=["disabled", "auto", "offline", "online"],
        default="disabled",
    )
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    args = parser.parse_args(argv)
    if args.hours <= 0 or args.hours > 3 or not math.isfinite(args.hours):
        parser.error("--hours must be positive and no greater than 3")
    if args.tasks != TASKS and (len(args.tasks) != 1 or args.tasks[0] not in TASKS):
        parser.error(
            "Use the approved five-task order, or exactly one task for integration"
        )
    return args


def main(argv=None):
    args = parse_args(argv)
    evaluation = Evaluation(args)

    def stop(signum, frame):
        evaluation.stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        if not args.worker and args.phase != "report":
            evaluation.start_reporter()
            # A separate supervisor bounds even stalled filesystem/network/library calls.
            # All nested workers, servers and containers inherit this worker's process group.
            command = [
                sys.executable,
                REPO / "terminalbench.py",
                args.phase,
                "--worker",
                "--resume",
                "--root",
                args.root,
                "--sft-run",
                args.sft_run,
                "--run-dir",
                args.run_dir,
                "--hours",
                str(args.hours),
                "--tasks",
                *args.tasks,
            ]
            try:
                evaluation.command(command, "evaluation-worker")
            finally:
                evaluation.state = read(evaluation.state_path)
                for record in [
                    *evaluation.state["trials"].values(),
                    *evaluation.state["controls"].values(),
                ]:
                    if record["status"] == "started":
                        record.update(
                            status="interrupted",
                            error="Supervisor shutdown; attempt will not be repeated",
                        )
        elif args.phase == "report":
            evaluation.report()
        else:
            if evaluation.state.get("error"):
                evaluation.state.setdefault("prior_errors", []).append(
                    evaluation.state.pop("error")
                )
            evaluation.state["status"] = "running"
            evaluation.save()
            if args.phase in ("all", "preflight") and not evaluation.state[
                "phases"
            ].get("preflight"):
                evaluation.preflight()
            if args.phase in ("all", "prepare") and not evaluation.state["phases"].get(
                "prepare"
            ):
                if args.phase == "all":
                    evaluation.verify_inputs()
                    evaluation.calibrate()
                evaluation.prepare()
            if args.phase in ("all", "calibrate", "run"):
                evaluation.run()
    except (Exception, KeyboardInterrupt) as exc:
        evaluation.state.update(
            status="partial"
            if isinstance(exc, (TimeoutError, KeyboardInterrupt))
            else "failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        evaluation.save()
        raise
    finally:
        evaluation.close()


if __name__ == "__main__":
    main()
