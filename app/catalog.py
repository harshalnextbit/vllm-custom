"""Authoritative model metadata for the inference engine.

Repository existence is verified from Hugging Face. ``verified`` is reserved
for variants that have passed this application's GPU, inference, streaming and
shutdown checks; candidates remain explicitly unverified until then.
"""

from __future__ import annotations

from copy import deepcopy
import re
from pathlib import Path
from typing import Any


def _variant(name: str, hf_id: str, *, quantization: str | None = None,
             dtype: str = "auto", trust_remote_code: bool = False,
             max_context: int = 8192, status: str = "candidate",
             reason: str | None = None,
             streaming_verified: bool = False,
             recommended_max_model_len: int | None = None,
             language_model_only: bool = False,
             enforce_eager: bool = False) -> dict[str, Any]:
    return {
        "variant": name,
        "hf_id": hf_id,
        "quantization": quantization,
        "architecture": None,
        "dtype": dtype,
        "recommended_vllm_quantization": quantization,
        "recommended_vllm_settings": {
            "max_model_len": recommended_max_model_len or min(max_context, 8192),
            "gpu_memory_utilization": 0.90,
            "enforce_eager": enforce_eager,
            "enable_prefix_caching": True,
            "tensor_parallel_size": 1,
            "trust_remote_code": trust_remote_code,
            "language_model_only": language_model_only,
        },
        "max_context": max_context,
        "max_output_tokens": 160,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.90,
        "enforce_eager": enforce_eager,
        "prefix_caching": True,
        "trust_remote_code": trust_remote_code,
        "status": status,
        "cached": False,
        "verified": status == "verified",
        "streaming_verified": streaming_verified,
        "reason": reason,
    }


SUPPORTED_MODELS: list[dict[str, Any]] = [
    {"display_name": "Sarvam-30B", "model_family": "sarvam", "max_context": 8192,
     "variants": [_variant("BF16", "sarvamai/sarvam-30b", dtype="bfloat16", trust_remote_code=True,
                           status="verified", streaming_verified=True, enforce_eager=True,
                           reason="Verified with isolated vLLM 0.17.0 CUDA 12.9 and Transformers 5.17.0; eager mode avoids a Transformers deprecation warning that Dynamo cannot trace."),
                  _variant("FP8", "sarvamai/sarvam-30b-fp8", quantization="fp8", dtype="auto", trust_remote_code=True,
                           reason="Unsupported on A100 SM80: vLLM ModelOpt FP8 requires compute capability 8.9 or newer.")]},
    {"display_name": "Qwen3-30B-A3B", "model_family": "qwen3", "max_context": 32768,
     "variants": [_variant("BF16", "Qwen/Qwen3-30B-A3B", dtype="bfloat16", status="verified",
                           streaming_verified=True, recommended_max_model_len=4096),
                  _variant("FP8", "Qwen/Qwen3-30B-A3B-FP8", quantization="fp8", status="verified",
                           streaming_verified=True, reason="Verified on A100 with vLLM 0.15.1 using weight-only FP8 via Marlin.")]},
    {"display_name": "Qwen3-14B", "model_family": "qwen3", "max_context": 32768,
     "variants": [_variant("BF16", "Qwen/Qwen3-14B", dtype="bfloat16",
                           max_context=32768, status="verified", streaming_verified=True,
                           recommended_max_model_len=8192)]},
    {"display_name": "Llama 3.3 70B", "model_family": "llama", "max_context": 8192,
     "variants": [_variant("BF16 (gated)", "meta-llama/Llama-3.3-70B-Instruct", dtype="bfloat16",
                           reason="Not tested: all supplied Hugging Face credentials received HTTP 403 for this repository, and the BF16 checkpoint exceeds this A100's 80 GB VRAM."),
                  _variant("AWQ", "casperhansen/llama-3.3-70b-instruct-awq", quantization="awq", dtype="float16",
                           status="verified", streaming_verified=True, recommended_max_model_len=4096)]},
    {"display_name": "Llama 3.2 3B", "model_family": "llama", "max_context": 8192,
     "variants": [_variant("BF16 (gated)", "meta-llama/Llama-3.2-3B-Instruct", dtype="bfloat16",
                           status="verified", streaming_verified=True,
                           reason="Verified using an authorized Hugging Face token; provide a token with repository access when downloading again."),
                  _variant("AWQ", "casperhansen/llama-3.2-3b-instruct-awq", quantization="awq", dtype="float16",
                           status="verified", streaming_verified=True)]},
    {"display_name": "Llama 3.1 8B Instruct", "model_family": "llama", "max_context": 8192,
     "variants": [_variant("BF16 (gated)", "meta-llama/Llama-3.1-8B-Instruct", dtype="bfloat16",
                           status="verified", streaming_verified=True,
                           reason="Verified with an authorized Hugging Face token; the token used for a future download must have Llama 3.1 access."),
                  _variant("AWQ", "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4", quantization="awq", dtype="float16",
                           status="verified", streaming_verified=True)]},
    {"display_name": "Gemma 3 27B IT", "model_family": "gemma3", "max_context": 8192,
     "variants": [_variant("BF16 (gated)", "google/gemma-3-27b-it", dtype="bfloat16", status="verified",
                           streaming_verified=True, reason="Verified using an authorized Hugging Face token; provide a token with Gemma access when downloading again.")]},
    {"display_name": "Phi-4 14B", "model_family": "phi4", "max_context": 8192,
     "variants": [_variant("BF16", "microsoft/phi-4", dtype="bfloat16", status="verified",
                           streaming_verified=True)]},
    {"display_name": "Qwen3.6-27B", "model_family": "qwen3.5", "max_context": 8192,
     "variants": [_variant("BF16", "Qwen/Qwen3.6-27B", dtype="bfloat16", status="verified",
                           streaming_verified=True, reason="Verified with isolated vLLM 0.17.0 CUDA 12.9 runtime and V1 model runner.",
                           recommended_max_model_len=4096, language_model_only=True),
                  _variant("FP8", "Qwen/Qwen3.6-27B-FP8", quantization="fp8", status="verified",
                           streaming_verified=True, reason="Verified with isolated vLLM 0.17.0 CUDA 12.9 runtime; weight-only FP8 via Marlin on A100.",
                           recommended_max_model_len=4096, language_model_only=True)]},
    {"display_name": "Qwen3.6-35B-A3B", "model_family": "qwen3.5-moe", "max_context": 8192,
     "variants": [_variant("BF16", "Qwen/Qwen3.6-35B-A3B", dtype="bfloat16", status="verified",
                           streaming_verified=True, reason="Verified with isolated vLLM 0.17.0 on A100; model used about 75 GiB VRAM.", language_model_only=True),
                  _variant("FP8", "Qwen/Qwen3.6-35B-A3B-FP8", quantization="fp8", status="verified",
                           streaming_verified=True, reason="Verified with isolated vLLM 0.17.0 CUDA 12.9 runtime and Marlin FP8 MoE backend on A100.",
                           recommended_max_model_len=4096, language_model_only=True)]},
    {"display_name": "Qwen3.8-27B", "model_family": "qwen3.5", "max_context": 8192,
     "variants": [_variant("BF16", "Qwen/Qwen3.8-27B", dtype="bfloat16", status="verified",
                           streaming_verified=True, reason="Verified with isolated vLLM 0.17.0 CUDA 12.9 runtime and V1 model runner.",
                           recommended_max_model_len=4096, language_model_only=True),
                  _variant("FP8", "Qwen/Qwen3.8-27B-FP8", quantization="fp8", status="verified",
                           streaming_verified=True, reason="Verified with isolated vLLM 0.17.0; A100 uses weight-only FP8 via Marlin.", language_model_only=True)]},
    {"display_name": "Gemma 4 26B-A4B-IT", "model_family": "gemma4", "max_context": 8192,
     "variants": [_variant("BF16", "google/gemma-4-26B-A4B-it", dtype="bfloat16",
                           status="verified", streaming_verified=True,
                           recommended_max_model_len=4096,
                           reason="Verified on A100 with vLLM 0.29.0 CUDA 12.9 wheel; normal and SSE inference passed. The CUDA 13.0 wheel is incompatible with the host's R550 driver.") ]},
    {"display_name": "DeepSeek-R1-Distill-Qwen-32B", "model_family": "deepseek", "max_context": 8192,
     "variants": [_variant("BF16", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", dtype="bfloat16", status="verified",
                           streaming_verified=True, reason="Verified on A100 with vLLM 0.15.1; reasoning model returned the requested marker with max_tokens=160."),
                  _variant("AWQ", "casperhansen/deepseek-r1-distill-qwen-32b-awq", quantization="awq", dtype="float16",
                           status="verified", streaming_verified=True,
                           reason="Verified on A100 with vLLM 0.15.1; reasoning model returned the requested marker with max_tokens=160.")]},
    {"display_name": "Mistral Small 3.2 24B", "model_family": "mistral", "max_context": 8192,
     "variants": [_variant("BF16", "mistralai/Mistral-Small-3.2-24B-Instruct-2506", dtype="bfloat16", status="verified",
                           streaming_verified=True)]},
    {"display_name": "Aya Expanse 32B", "model_family": "aya", "max_context": 8192,
     "variants": [_variant("BF16 (gated)", "CohereLabs/aya-expanse-32b", dtype="bfloat16",
                           reason="Not tested: all three supplied Hugging Face credentials received HTTP 403 for this gated repository."),
                  _variant("AWQ", "Orion-zhen/aya-expanse-32b-AWQ", quantization="awq", dtype="float16",
                           status="verified", streaming_verified=True)]},
]


def _estimate_model_weights_gb(model_id: str | None, quantization: str | None = None) -> float | None:
    if not model_id:
        return None
    mid = model_id.lower()
    quant = (quantization or "").lower()
    is_4bit = "awq" in quant or "gptq" in quant or "4bit" in quant or "awq" in mid

    match = re.search(r"(\d+(?:\.\d+)?)\s*b", mid)
    if match:
        params = float(match.group(1))
        bytes_per_param = 0.58 if is_4bit else 2.05
        return round((params * 1e9 * bytes_per_param) / (1024 ** 3), 2)

    return 3.10

def get_catalog() -> list[dict[str, Any]]:
    catalog = deepcopy(SUPPORTED_MODELS)
    for family in catalog:
        for variant in family.get("variants", []):
            hf_id = variant.get("hf_id")
            quant = variant.get("quantization")
            size_gb = _estimate_model_weights_gb(hf_id, quant)
            variant["model_size"] = f"{size_gb} GB" if size_gb else "Unknown"
        
        if family.get("variants"):
            family["model_size"] = family["variants"][0]["model_size"]
        else:
            family["model_size"] = "Unknown"
    return catalog


def get_variant(model_id: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    for family in SUPPORTED_MODELS:
        for variant in family["variants"]:
            if variant["hf_id"].casefold() == model_id.casefold():
                return family, deepcopy(variant)
    return None, None


def discover_cached_models(cache_dir: str | None = None) -> list[dict[str, Any]]:
    """Return Hub repositories with at least one completed model-weight file.

    Snapshot folders can contain only configs/tokenizers (for example after a
    failed download). Those repositories must not bypass disk preflight or be
    presented as ready-to-load cached models.
    """
    try:
        from huggingface_hub import scan_cache_dir
        cache = scan_cache_dir(cache_dir) if cache_dir else scan_cache_dir()
        cached = []
        for repo in cache.repos:
            if repo.repo_type != "model" or not repo.repo_path:
                continue
            weight_suffixes = (".safetensors", ".bin", ".pt", ".pth", ".model", ".gguf")
            cached_weight_files = [
                file.file_name
                for revision in repo.revisions
                for file in revision.files
                if file.size_on_disk > 0 and file.file_name.lower().endswith(weight_suffixes)
            ]
            # Newer Hugging Face Hub runtimes can leave only some numbered
            # shards in the legacy snapshot index after an interrupted load.
            # Do not call that repository cached: vLLM would skip disk
            # preflight and then fail trying to resolve the missing shards.
            shard_groups: dict[str, tuple[int, set[int]]] = {}
            for name in cached_weight_files:
                match = re.search(r"(?P<prefix>.*?)(?P<index>\d+)-of-(?P<total>\d+)(?P<suffix>\.[^.]+)$", name, re.IGNORECASE)
                if not match:
                    continue
                key = f"{match.group('prefix')}of-{match.group('total')}{match.group('suffix')}"
                total, indices = shard_groups.setdefault(key, (int(match.group("total")), set()))
                indices.add(int(match.group("index")))
            has_weights = bool(cached_weight_files) and all(
                len(indices) == total for total, indices in shard_groups.values()
            )
            if cached_weight_files and shard_groups and not has_weights:
                # huggingface_hub 1.x may store completed large files in its
                # shared, content-addressed blob tree while only some snapshot
                # links are materialized. Its small .refs files tie each blob
                # back to the repository and let us distinguish a full cached
                # set from config-only or interrupted downloads.
                shared_blob_root = Path(repo.repo_path).parent / "blobs"
                repo_cache_key = "models--" + repo.repo_id.replace("/", "--")
                referenced_hashes: set[str] = set()
                for ref_file in shared_blob_root.glob("*/*.refs"):
                    try:
                        lines = ref_file.read_text(errors="ignore").splitlines()
                        for line in lines:
                            prefix = repo_cache_key + "/blobs/"
                            if line.startswith(prefix):
                                blob = ref_file.with_suffix("")
                                if blob.is_file() and blob.stat().st_size > 0:
                                    referenced_hashes.add(line[len(prefix):])
                    except OSError:
                        continue
                expected_shards = sum(total for total, _ in shard_groups.values())
                has_weights = len(referenced_hashes) >= expected_shards
            if not has_weights:
                continue
            family, variant = get_variant(repo.repo_id)
            cached.append({
                "hf_id": repo.repo_id,
                "cached": True,
                "custom": family is None,
                "verified": bool(variant and variant.get("verified")),
                "family": family["display_name"] if family else None,
                "variant": variant["variant"] if variant else None,
                "path": repo.repo_path,
            })
        return sorted(cached, key=lambda item: item["hf_id"].casefold())
    except Exception:
        return []


def catalog_payload(active_model: str | None = None) -> dict[str, Any]:
    cached = discover_cached_models()
    cached_ids = {item["hf_id"].casefold() for item in cached}
    catalog = get_catalog()
    for family in catalog:
        for variant in family["variants"]:
            variant["cached"] = variant["hf_id"].casefold() in cached_ids
            variant["active"] = variant["hf_id"] == active_model
    return {"supported_models": catalog,
            "custom_cached_models": [item for item in cached if item["custom"]],
            "cached_models": [item["hf_id"] for item in cached]}
