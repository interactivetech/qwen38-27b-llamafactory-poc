"""GPU preparation runs under the unchanged training interpreter, never vLLM's."""

import gc
import hashlib
import sys
import warnings
from pathlib import Path

from terminalbench import gpu_free, hashes, read, sha, verify_checkpoint, write
from terminalbench_qwen import TEMPLATE_KWARGS

PROBES = [
    (
        [
            {
                "role": "user",
                "content": "What is two plus two? Answer in one short sentence.",
            }
        ],
        "The result is four.",
    ),
    (
        [{"role": "user", "content": "Name a primary color."}],
        "Blue is a primary color.",
    ),
    ([{"role": "user", "content": "Complete this sequence: a, b,"}], " c."),
    (
        [
            {"role": "user", "content": "Name a shape."},
            {
                "role": "assistant",
                "content": "A square.",
                "reasoning_content": "A square has four equal sides.",
            },
            {"role": "user", "content": "How many sides does that shape have?"},
        ],
        "Four sides.",
    ),
]

# A training cache intentionally omits model-card/license/media files. Restrict
# Hub's snapshot-completeness check to text inference assets; shard integrity
# remains independently checked against the pinned remote index and metadata.
INFERENCE_FILES = [
    "config.json",
    "generation_config.json",
    "model*.safetensors",
    "model.safetensors.index.json",
    "tokenizer*.json",
    "chat_template.jinja",
    "merges.txt",
    "vocab.json",
    "special_tokens_map.json",
]


def cached_base(revision):
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            "Qwen/Qwen3.8-27B",
            revision=revision,
            allow_patterns=INFERENCE_FILES,
            local_files_only=True,
        )
    )


def verify_adapter_tensors(saved, loaded):
    """Reject partial loading, including freshly initialized LoRA modules."""
    import torch

    if not saved or set(saved) != set(loaded):
        raise ValueError("Adapter tensor keys do not match the evaluation model")
    for key, expected in saved.items():
        observed = loaded[key].detach().cpu()
        if observed.shape != expected.shape or not torch.equal(observed, expected):
            raise ValueError(f"Adapter tensor did not load exactly: {key}")


def load_verified_adapter(model, checkpoint):
    from peft import PeftModel, get_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights

    # AutoModelForCausalLM selects the text-only class from the multimodal base.
    # Training saved adapters under the enclosing conditional-generation class.
    mapping = (
        {r"^model\.language_model\.": "model."}
        if model.config.model_type == "qwen3_5_text"
        else {}
    )
    saved = load_peft_weights(str(checkpoint), device="cpu", key_mapping=mapping)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Found missing adapter keys.*")
        model = PeftModel.from_pretrained(
            model, checkpoint, is_trainable=False, key_mapping=mapping
        ).eval()
    loaded = get_peft_model_state_dict(model)
    verify_adapter_tensors(saved, loaded)
    return model, dict(tensor_count=len(saved), key_mapping=mapping, exact_match=True)


def main(run_dir):
    import importlib.metadata as metadata

    import torch
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run_dir = Path(run_dir)
    state = read(run_dir / "state.json")
    provenance = read(run_dir / "provenance.json")
    checkpoint, _ = verify_checkpoint(Path(state["sft_run"]))
    gpu_free()
    if (
        metadata.version("transformers") != "5.17.0"
        or metadata.version("peft") != "0.18.1"
    ):
        raise ValueError(
            "Reference environment changed; requires Transformers 5.17.0 and PEFT 0.18.1"
        )
    base = cached_base(provenance["model_revision"])
    if base.name != provenance["model_revision"]:
        raise ValueError("Base snapshot revision mismatch")
    remote = HfApi().model_info(
        "Qwen/Qwen3.8-27B", revision=provenance["model_revision"], files_metadata=True
    )
    if remote.sha != provenance["model_revision"]:
        raise ValueError("Resolved model revision mismatch")
    base_hashes = {}
    for file in remote.siblings:
        path = base / file.rfilename
        if not path.is_file():
            continue
        if file.lfs:
            digest = sha(path)
            if digest != file.lfs.sha256:
                raise ValueError(
                    f"Base model cache integrity failure: {file.rfilename}"
                )
        else:
            data = path.read_bytes()
            git_blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            if git_blob != file.blob_id:
                raise ValueError(f"Base model file changed: {file.rfilename}")
            digest = hashlib.sha256(data).hexdigest()
        base_hashes[file.rfilename] = digest
    index = read(base / "model.safetensors.index.json")
    if not index or not set(index["weight_map"].values()).issubset(base_hashes):
        raise ValueError("Base model snapshot is missing weight shards")
    if (
        read(checkpoint / "adapter_config.json")["base_model_name_or_path"]
        != "Qwen/Qwen3.8-27B"
    ):
        raise ValueError("Adapter was trained for a different base model")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    template = (checkpoint / "chat_template.jinja").read_text()
    if tokenizer.chat_template != template:
        raise ValueError("Saved tokenizer/template mismatch")
    # Both servers use this exact tokenizer/template, including the base server.
    probes = []
    for messages, answer in PROBES:
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            **TEMPLATE_KWARGS,
        )
        if (
            not isinstance(ids, list)
            or not ids
            or not all(isinstance(token, int) for token in ids)
        ):
            raise ValueError(
                "Reference tokenizer did not return a flat prompt-token list"
            )
        target = tokenizer.encode(answer, add_special_tokens=False)
        probes.append(dict(messages=messages, prompt=ids, target=target))

    def values(model):
        result = []
        for probe in probes:
            ids = probe["prompt"] + probe["target"]
            inputs = torch.tensor([ids], device="cuda")
            with torch.inference_mode():
                logits = model(inputs).logits[0].float()
            offset = len(probe["prompt"])
            lp = logits[offset - 1 : len(ids) - 1].log_softmax(-1)
            chosen = lp.gather(1, inputs[0, offset:].unsqueeze(1)).squeeze(1)
            result.append(chosen.cpu().tolist())
        return result

    model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    base_values = values(model)
    model, adapter_loading = load_verified_adapter(model, checkpoint)
    write(run_dir / "adapter-loading.json", adapter_loading)
    adapter_values = values(model)
    model = model.merge_and_unload(safe_merge=True).eval()
    merged_values = values(model)
    differences = [
        abs(sum(a) / len(a) - sum(b) / len(b))
        for a, b in zip(adapter_values, merged_values)
    ]
    mean_difference = sum(differences) / len(differences)
    write(
        run_dir / "merge-equivalence.json",
        dict(
            base=base_values,
            adapter=adapter_values,
            merged=merged_values,
            mean_nll_difference=mean_difference,
            probe_differences=differences,
            tolerance=0.01,
            within_tolerance=mean_difference <= 0.01,
            policy="advisory",
        ),
    )
    if mean_difference > 0.01:
        print(
            f"Numerical advisory: merge chosen-token NLL difference {mean_difference} "
            "exceeds historical tolerance 0.01; continuing per user instruction.",
            flush=True,
        )
    output = run_dir / "merged-bf16"
    if output.exists():
        raise ValueError(f"Refusing to overwrite merged copy: {output}")
    output.mkdir()
    model.save_pretrained(output, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(output)
    (output / "chat_template.jinja").write_text(template)
    if any(
        p.dtype != torch.bfloat16 for p in model.parameters() if p.is_floating_point()
    ):
        raise ValueError("Merged parameters are not all BF16")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    # Check serialized weights too, catching a corrupt or incomplete evaluation copy.
    reloaded = AutoModelForCausalLM.from_pretrained(
        output,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    reload_values = values(reloaded)
    if (
        max(
            abs(x - y)
            for a, b in zip(merged_values, reload_values)
            for x, y in zip(a, b)
        )
        > 1e-5
    ):
        raise ValueError("Serialized merged model changed probe values")
    write(
        run_dir / "reference.json",
        dict(
            probes=probes,
            base=base_values,
            sft=reload_values,
            merge_mean_nll_difference=mean_difference,
            merge_probe_differences=differences,
        ),
    )
    write(
        run_dir / "prepared.json",
        dict(
            base=str(base),
            base_hashes=base_hashes,
            sft=str(output),
            tokenizer=str(output),
            merged_hashes=hashes(output),
            checkpoint_hashes=hashes(checkpoint),
            tokenizer_hashes={
                k: v
                for k, v in hashes(output).items()
                if "tokenizer" in k or "template" in k
            },
            reference_sha256=hashes(run_dir, names=["reference.json"])[
                "reference.json"
            ],
            adapter_loading=adapter_loading,
        ),
    )


if __name__ == "__main__":
    main(sys.argv[1])
