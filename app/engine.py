"""Serialized lifecycle manager for an isolated vLLM OpenAI server process."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import sys
import time
import shutil
from pathlib import Path
from typing import Any

import httpx
import torch

from app.catalog import discover_cached_models, get_variant
from app.config import get_settings


class VLLMManager:
    _instance: "VLLMManager | None" = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._lock = asyncio.Lock()
        self.process: asyncio.subprocess.Process | None = None
        self.model_id: str | None = None
        self.active_config: dict[str, Any] = {}
        self.status = "STOPPED"
        self.error: str | None = None
        self.started_at: float | None = None
        self.port: int | None = None
        self.log_path: str | None = None
        self.gpu_free_before_start: int | None = None

    @classmethod
    def reset(cls):
        cls._instance = None

    def is_loaded(self) -> bool:
        return self.status == "READY" and self.process is not None and self.process.returncode is None

    @property
    def api_base_url(self) -> str | None:
        return f"http://127.0.0.1:{self.port}/v1" if self.port else None

    def status_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "model_id": self.model_id,
            "pid": self.process.pid if self.process and self.process.returncode is None else None,
            "port": self.port,
            "gpu": os.getenv("CUDA_VISIBLE_DEVICES", "0"),
            "started_at": self.started_at,
            "active_config": dict(self.active_config),
            "error": self.error,
            "log_tail": self._tail_log(),
        }

    def _tail_log(self, limit: int = 5000) -> str:
        if not self.log_path:
            return ""
        try:
            with open(self.log_path, "rb") as log:
                log.seek(max(0, os.path.getsize(self.log_path) - limit))
                return log.read().decode("utf-8", errors="replace")
        except OSError:
            return ""

    @staticmethod
    def _free_port() -> int:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    async def load_model(self, model_id: str, quantization: str | None = None,
                         max_model_len: int | None = None,
                         gpu_memory_utilization: float | None = None,
                         enforce_eager: bool | None = None,
                         hf_token: str | None = None,
                         dtype: str | None = None,
                         tensor_parallel_size: int | None = None,
                         trust_remote_code: bool | None = None,
                         enable_prefix_caching: bool | None = None) -> dict[str, Any]:
        async with self._lock:
            settings = get_settings()
            family, variant = get_variant(model_id)
            if variant:
                defaults = variant["recommended_vllm_settings"]
                quantization = quantization if quantization not in (None, "none", "") else variant["recommended_vllm_quantization"]
                max_model_len = max_model_len or defaults["max_model_len"]
                gpu_memory_utilization = gpu_memory_utilization or defaults["gpu_memory_utilization"]
                enforce_eager = defaults["enforce_eager"] if enforce_eager is None else enforce_eager
                dtype = dtype or variant["dtype"]
                tensor_parallel_size = tensor_parallel_size or defaults["tensor_parallel_size"]
                trust_remote_code = defaults["trust_remote_code"] if trust_remote_code is None else trust_remote_code
                enable_prefix_caching = defaults["enable_prefix_caching"] if enable_prefix_caching is None else enable_prefix_caching
                language_model_only = defaults.get("language_model_only", False)
            else:
                max_model_len = max_model_len or settings.MAX_MODEL_LEN
                gpu_memory_utilization = gpu_memory_utilization or settings.GPU_MEMORY_UTILIZATION
                enforce_eager = settings.ENFORCE_EAGER if enforce_eager is None else enforce_eager
                dtype = dtype or settings.DTYPE
                tensor_parallel_size = tensor_parallel_size or settings.TENSOR_PARALLEL_SIZE
                trust_remote_code = settings.TRUST_REMOTE_CODE if trust_remote_code is None else trust_remote_code
                enable_prefix_caching = settings.ENABLE_PREFIX_CACHING if enable_prefix_caching is None else enable_prefix_caching
                language_model_only = False

            if self.model_id == model_id and self.is_loaded():
                return self.status_payload()

            self.status = "STOPPING"
            await self._stop_process()
            self.model_id = model_id
            self.error = None
            self.status = "STARTING"
            port = self._free_port()
            self.port = port
            self.started_at = time.time()
            self.active_config = {
                "model_id": model_id, "quantization": quantization or "auto",
                "dtype": dtype, "max_model_len": max_model_len,
                "gpu_memory_utilization": gpu_memory_utilization,
                "enforce_eager": enforce_eager,
                "tensor_parallel_size": tensor_parallel_size,
                "enable_prefix_caching": enable_prefix_caching,
                "trust_remote_code": trust_remote_code,
                "language_model_only": language_model_only,
            }
            runtime_key = family["model_family"] if family else "default"
            runtime_python = settings.VLLM_RUNTIME_MAP.get(runtime_key)
            runtime_python = runtime_python or settings.VLLM_RUNTIME_MAP.get("default") or sys.executable
            if not Path(runtime_python).is_file():
                self.status = "FAILED"
                self.error = f"Configured vLLM runtime Python does not exist: {runtime_python}"
                raise RuntimeError(self.error)
            self.active_config["runtime_python"] = runtime_python

            log_dir = Path(settings.LOG_DIR or "./logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            safe_name = model_id.replace("/", "_").replace(":", "_")
            self.log_path = str(log_dir / f"vllm-{safe_name}-{int(self.started_at)}.log")

            args = [runtime_python, "-m", "vllm.entrypoints.openai.api_server",
                    "--model", model_id, "--host", "127.0.0.1", "--port", str(port),
                    "--dtype", str(dtype), "--max-model-len", str(max_model_len),
                    "--gpu-memory-utilization", str(gpu_memory_utilization),
                    "--tensor-parallel-size", str(tensor_parallel_size),
                    "--served-model-name", model_id,
                    "--trust-request-chat-template"]
            # Stelterlab's Mistral Small 3.2 FP8 checkpoint is stored in
            # Mistral's consolidated format and uses the Mistral3 config. The
            # generic HF loader can resolve its text config as MistralForCausalLM
            # and then fails on the checkpoint's vision/projector tensors.
            if model_id.casefold() == "stelterlab/mistral-small-3.2-24b-instruct-2506-fp8":
                args += ["--tokenizer-mode", "mistral",
                         "--config-format", "mistral",
                         "--load-format", "mistral"]
            if quantization and quantization.lower() not in ("none", "auto"):
                args += ["--quantization", quantization.lower()]
            if enforce_eager:
                args.append("--enforce-eager")
            if trust_remote_code:
                args.append("--trust-remote-code")
            if enable_prefix_caching:
                args.append("--enable-prefix-caching")
            if language_model_only:
                args.append("--language-model-only")
            if settings.CACHE_DIR:
                args += ["--download-dir", settings.CACHE_DIR]

            env = os.environ.copy()
            # The pinned vLLM 0.29 runtime defaults to Model Runner V2, whose
            # pinned-host-memory UVA registration requires a newer CUDA driver
            # than the host currently provides.  Keep the newer model
            # architectures on this isolated runtime while selecting its
            # compatible V1 runner path.
            if Path(runtime_python).absolute() != Path(sys.executable).absolute():
                env.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
            runtime_lib = Path(runtime_python).absolute().parent.parent / "lib"
            runtime_python_sites = sorted(runtime_lib.glob("python*/site-packages/nvidia"))
            runtime_site = runtime_python_sites[0] if runtime_python_sites else runtime_lib / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "nvidia"
            runtime_lib_dirs = [runtime_site / "cu13" / "lib", runtime_site / "cuda_runtime" / "lib"]
            runtime_path = str(Path(runtime_python).absolute())
            if "vllm-0.17.0" in runtime_path or "vllm-0.29.0" in runtime_path:
                # The CUDA 12.9 runtimes install their libraries as separate
                # nvidia/* packages. The 0.29 environment also retains the
                # CUDA 13 package, so exclude it to keep CUDA 12.9 libraries
                # ahead of the host's older driver.
                runtime_lib_dirs = [
                    path for path in runtime_site.glob("*/lib")
                    if "vllm-0.29.0" not in runtime_path or path.parent.name != "cu13"
                ]
            existing_ld_path = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = ":".join(
                [str(path) for path in runtime_lib_dirs if path.is_dir()] + ([existing_ld_path] if existing_ld_path else [])
            )
            token = hf_token or settings.HF_TOKEN
            if token:
                env["HF_TOKEN"] = token
                env["HUGGING_FACE_HUB_TOKEN"] = token
            try:
                await self._check_disk_space(model_id, token)
                if torch.cuda.is_available():
                    try:
                        self.gpu_free_before_start, _ = torch.cuda.mem_get_info(0)
                    except Exception:
                        self.gpu_free_before_start = None
                with open(self.log_path, "ab", buffering=0) as log:
                    self.process = await asyncio.create_subprocess_exec(
                        *args, stdout=log, stderr=log, env=env,
                        start_new_session=True,
                    )
                await self._wait_ready(model_id)
                self.status = "WARMING_UP"
                async with httpx.AsyncClient(timeout=180.0) as client:
                    response = await client.post(f"{self.api_base_url}/chat/completions", json={
                        "model": model_id,
                        "messages": [{"role": "user", "content": "Reply with one short greeting."}],
                        "max_tokens": 12,
                        "temperature": 0,
                    })
                    response.raise_for_status()
                    choices = response.json().get("choices") or []
                    if not choices or not str(choices[0].get("message", {}).get("content", "")).strip():
                        raise RuntimeError("Warm-up completion returned empty content")
                self.status = "READY"
                return self.status_payload()
            except Exception as exc:
                logs = self._tail_log()
                self.error = f"{type(exc).__name__}: {exc}" + (f"\n{logs[-3000:]}" if logs else "")
                self.status = "FAILED"
                await self._stop_process()
                self.model_id = model_id
                self.status = "FAILED"
                raise RuntimeError(self.error) from exc

    async def _wait_ready(self, expected_model: str, timeout: float = 1800) -> None:
        deadline = time.monotonic() + timeout
        last_error = ""
        async with httpx.AsyncClient(timeout=8.0) as client:
            while time.monotonic() < deadline:
                if self.process is None or self.process.returncode is not None:
                    raise RuntimeError(f"vLLM server exited with code {self.process.returncode if self.process else 'unknown'}")
                try:
                    health = await client.get(f"http://127.0.0.1:{self.port}/health")
                    models = await client.get(f"http://127.0.0.1:{self.port}/v1/models")
                    served = [item.get("id") for item in models.json().get("data", [])]
                    if health.status_code == 200 and expected_model in served:
                        return
                except Exception as exc:
                    last_error = str(exc)
                await asyncio.sleep(2)
        raise TimeoutError(f"vLLM did not become ready within {timeout}s. {last_error}")

    async def _check_disk_space(self, model_id: str, token: str | None) -> None:
        settings = get_settings()
        cache_dir = Path(settings.CACHE_DIR or os.getenv("HF_HOME", Path.home() / ".cache/huggingface"))
        try:
            cache_state = await asyncio.to_thread(discover_cached_models)
            if any(item["hf_id"].casefold() == model_id.casefold() for item in cache_state):
                return
            from huggingface_hub import HfApi
            info = await asyncio.to_thread(HfApi(token=token).model_info, model_id, files_metadata=True)
            estimate = 0
            for sibling in info.siblings or []:
                name = sibling.rfilename.lower()
                if name.endswith((".safetensors", ".bin", ".pt", ".model", ".gguf")):
                    size = getattr(sibling, "size", None)
                    lfs = getattr(sibling, "lfs", None)
                    size = size or (getattr(lfs, "size", None) if lfs else None)
                    if size:
                        estimate += int(size)
            if estimate:
                free = shutil.disk_usage(cache_dir.parent if cache_dir.name == "huggingface" else cache_dir).free
                safety_margin = 5 * 1024**3
                if free < estimate + safety_margin:
                    raise OSError(
                        f"Insufficient disk space for {model_id}: estimated {estimate / 1024**3:.1f} GiB, "
                        f"available {free / 1024**3:.1f} GiB (requires 5 GiB safety margin)."
                    )
        except OSError:
            raise
        except Exception:
            # Do not block an offline, already cached model due to metadata lookup.
            # vLLM/Hugging Face still reports repository, auth and download failures.
            return

    async def unload_model(self) -> dict[str, Any]:
        async with self._lock:
            previous = self.model_id
            self.status = "STOPPING"
            await self._stop_process()
            self.model_id = None
            self.active_config = {}
            self.error = None
            self.status = "STOPPED"
            self.port = None
            return {"status": "STOPPED", "previous_model": previous}

    async def _stop_process(self) -> None:
        proc = self.process
        port = self.port
        free_before_stop = self.gpu_free_before_start
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if proc is not None and proc.returncode is None:
            try:
                await asyncio.wait_for(proc.wait(), timeout=30)
            except asyncio.TimeoutError:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await proc.wait()

        if proc is not None:
            group_deadline = time.monotonic() + 10
            while time.monotonic() < group_deadline:
                try:
                    os.killpg(proc.pid, 0)
                except ProcessLookupError:
                    break
                await asyncio.sleep(0.25)
            else:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                raise TimeoutError(f"vLLM process group {proc.pid} did not terminate")

        self.process = None
        self.port = None
        await self._wait_gpu_release(free_before_stop, timeout=45)
        self.gpu_free_before_start = None
        if port is not None:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                with socket.socket() as sock:
                    if sock.connect_ex(("127.0.0.1", port)) != 0:
                        break
                await asyncio.sleep(0.25)

    async def _wait_gpu_release(self, free_before_stop: int | None, timeout: float) -> None:
        if not torch.cuda.is_available() or free_before_stop is None:
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            await asyncio.sleep(1)
            try:
                free, _ = torch.cuda.mem_get_info(0)
                # Compare against the free memory from before launch. Comparing
                # against memory immediately before shutdown falsely reports a
                # leak when startup failed before allocating any GPU memory.
                # Allow small background allocations to vary by 256 MiB.
                if free >= free_before_stop - 256 * 1024 * 1024:
                    return
            except Exception:
                return
        raise TimeoutError("GPU memory did not recover after stopping the vLLM process")

    async def proxy_request(self, path: str, payload: dict[str, Any], stream: bool = False):
        if not self.is_loaded() or not self.api_base_url:
            raise RuntimeError("No model is ready")
        url = f"{self.api_base_url}/{path.lstrip('/')}"
        if stream:
            return url
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()


def get_vllm_manager() -> VLLMManager:
    return VLLMManager()
