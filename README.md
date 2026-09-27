# Production vLLM FastAPI Inference Engine & Cloudflare Quick Tunnel

A lean, high-throughput, model-agnostic LLM Inference Engine powered strictly by **vLLM (AsyncLLMEngine >= 0.6.0)** and **FastAPI**.

Designed for instant deployment in **Google Colab** (NVIDIA T4, L4, A100) or self-hosted GPU servers, exposed publicly via **Cloudflare Quick Tunnel (`trycloudflare.com`)** or run strictly on **localhost** via environment toggles.

---

## ⚡ Core Architecture

- **Zero Startup Lag (No Default Model)**: Starts in milliseconds in standby mode (`status: "READY_NO_MODEL"`). Immediately prints the public Cloudflare Quick Tunnel URL and Swagger UI (`/docs`).
- **Dynamic Model Lifecycle**: Hot-swap or load models on the fly via `POST /admin/load-model` and release all VRAM back to 0 MB with `POST /admin/unload-model`.
- **PagedAttention & Continuous Batching**: High-concurrency generation with chunked prefill, RadixAutomatic Prefix Caching (APC), and Server-Sent Events (SSE).
- **DeepSeek Reasoning First-Class Support**: Native parsing of `<think>` and `</think>` chain-of-thought tokens for DeepSeek-R1 and QwQ models.
- **100% Configurable via `.env`**: Network ports, CORS, memory fractions, context windows, and fallback behaviors.
- **Enterprise Security (VAPT-Audited)**: Path traversal rejection, finite numeric bounds, secret redaction, and defensive HTTP security headers (`nosniff`, `DENY`, `strict-origin`).
- **Portable Web UI**: Standalone React + Lucide React + Tailwind SPA located in `frontend/`.

---

### Isolated model runtimes

Set `VLLM_RUNTIME_MAP` in `.env` to route dependency-sensitive model families to their own Python environments. The current A100 configuration uses vLLM 0.17 for Qwen3.5, vLLM 0.17 with Transformers 5 for Sarvam, and vLLM 0.29 with the CUDA 12.9 wheel for Gemma 4. The CUDA 13.0 vLLM 0.29 wheel fails on the host's R550 driver; keep CUDA 12.9 libraries ahead of any leftover CUDA 13 libraries in `LD_LIBRARY_PATH`. Gemma 4 has now passed model warm-up plus normal and SSE inference with this setup.

The catalog's `verified` and `streaming_verified` flags reflect successful model startup and real responses from both API modes. A verified model still needs enough local disk and GPU memory to load; the UI can select it and the backend starts its configured runtime when requested.

## 🚀 Quick Start

### Option A: Google Colab (One-Click Launch)

**Method 1: Direct Zip Upload (Recommended — No Token Needed)**
1. Zip this project folder locally: `zip -r vllm-engine.zip . -x "node_modules/*" ".git/*"`
2. Upload `vllm-engine.zip` to Colab and execute:
```bash
!unzip -q vllm-engine.zip -d vllm-inference-engine
%cd vllm-inference-engine
!sed -i 's/\r$//' run_colab.sh && bash run_colab.sh
```

**Method 2: Private GitHub Repository Clone (via `.env`)**
If cloning directly from your private GitHub repository in Colab, configure your `.env`:
```ini
GITHUB_TOKEN=ghp_yourPersonalAccessToken
REPO_URL=https://github.com/your-username/vllm-inference-engine.git
```
Then run:
```bash
!sed -i 's/\r$//' run_colab.sh && bash run_colab.sh
```
`run_colab.sh` reads `GITHUB_TOKEN` and `REPO_URL` directly from `.env`, clones the private repo, starts the Cloudflare Quick Tunnel, and boots the engine.

When running, it automatically starts the Cloudflare Quick Tunnel and outputs:
```text
==================================================================
 🌐 Cloudflare Quick Tunnel Live: https://xxx-yyy-zzz.trycloudflare.com
 📚 Swagger UI Documentation:    https://xxx-yyy-zzz.trycloudflare.com/docs
 💻 Local Address:               http://localhost:8006
==================================================================
```

### Option B: Local or Self-Hosted GPU
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and customize configuration
cp .env.example .env

# 3. Start server (starts idle on localhost)
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8006
```

---

## Model registry and runtime selection

`app/catalog.py` is the model catalog source of truth. `GET /admin/catalog` returns verified entries, unverified candidates, cached catalog variants, and custom cached repositories. The frontend consumes that API instead of keeping its own model list. A candidate is not marked verified until isolated GPU load, inference, streaming, and shutdown have been recorded.

The API starts an isolated `vllm serve` child process for each selected model, waits for a real warm-up completion, and uses the same OpenAI-compatible process for inference. Custom Hugging Face repositories use this same lifecycle. Cache discovery and disk checks use Hugging Face metadata; stop/unload terminates the process group and checks GPU memory recovery.

The engine serves one selected model at a time. For chat and completion requests, it replaces the client's `model` field with the active model ID. This lets clients such as LiveKit keep a static startup model setting while the UI switches the backend to a different catalog model.

### Isolated architecture runtimes

The base API runtime remains unchanged. `VLLM_RUNTIME_MAP` selects a separate Python environment per model family:

- **Qwen3.6/Qwen3.5-family:** vLLM 0.17.0 with CUDA 12.9 wheels, installed by `bash scripts/setup_vllm_017_runtime.sh`. Qwen3.6-27B BF16 passed the end-to-end GPU, response, SSE, and shutdown checks on the current A100/driver when using the vLLM V1 runner.
- **Sarvam-30B:** vLLM 0.17.0 with CUDA 12.9 and Transformers 5.17, installed in a separate environment by `bash scripts/setup_vllm_017_transformers5_runtime.sh`. The BF16 variant passed with eager execution enabled. This isolated environment overrides vLLM's declared Transformers `<5` dependency; keep it separate from Qwen's Transformers 4 runtime.
- **Gemma 4:** vLLM 0.29.0 with Transformers 5.17, installed by `bash scripts/setup_vllm_029_runtime.sh`. The model loaded, but warm-up inference failed in a CUDA MoE kernel because the current host driver is too old for its CUDA runtime.

The launcher adds each runtime's CUDA library directories only to that model subprocess. Keep the matching entries in `.env` and use a driver/runtime combination supported by the GPU host. A catalog candidate is selectable but is not represented as verified until it passes GPU load, inference, streaming, and process cleanup tests.

### Current model validation status

On the NVIDIA A100 80 GB host, 21 of the catalog's 25 variants have passed isolated load, non-streaming inference, SSE through `[DONE]`, and shutdown. Four remain unverified or unsupported:

| Model variant | Hugging Face repository | Current status |
|---|---|---|
| Sarvam-30B FP8 | `sarvamai/sarvam-30b-fp8` | **Unsupported on this A100.** ModelOpt FP8 requires compute capability 8.9; this A100 is SM80. |
| Llama 3.3 70B BF16 | `meta-llama/Llama-3.3-70B-Instruct` | **Not run.** All supplied tokens received HTTP 403, and the BF16 weights exceed this GPU's 80 GB VRAM. |
| Gemma 4 26B-A4B IT BF16 | `google/gemma-4-26B-A4B-it` | **Unverified on this host.** vLLM 0.29 loaded the weights but warm-up inference failed in a CUDA kernel because the host driver is too old for its CUDA runtime. |
| Aya Expanse 32B BF16 | `CohereLabs/aya-expanse-32b` | **Not run.** All supplied tokens received HTTP 403 for this gated repository. |

The 21 variants that passed are Sarvam-30B BF16; Qwen3-30B-A3B BF16 and FP8; Qwen3-14B BF16; Llama 3.3 70B AWQ; Llama 3.2 3B BF16 and AWQ; Llama 3.1 8B BF16 and AWQ; Gemma 3 27B IT BF16; Phi-4 14B BF16; Qwen3.6-27B BF16 and FP8; Qwen3.6-35B-A3B BF16 and FP8; Qwen3.8-27B BF16 and FP8; DeepSeek-R1-Distill-Qwen-32B BF16 and AWQ; Mistral Small 3.2 24B BF16; and Aya Expanse 32B AWQ.

**Using a verified model:** Select a verified catalog entry and start inference once its load completes and the engine reports ready. Qwen3.8-27B was rechecked through the active port 8006 API: the catalog selected its isolated vLLM 0.17 runtime, normal and streamed inference returned `MODEL_TEST_OK`, and unload returned the engine to STOPPED. Six verified models currently have complete cached weights: Llama 3.2 3B BF16, Llama 3.2 3B AWQ, Llama 3.1 8B AWQ, Phi-4, Qwen3-14B, and Qwen3.8-27B. Other verified entries download when selected; gated models require a token with access to that repository.

## 🎛️ How to Load Models

### 1. Via Swagger UI (`/docs`)
1. Open the public Cloudflare URL: `https://<tunnel-id>.trycloudflare.com/docs`.
2. Expand `POST /admin/load-model`.
3. Paste your desired model configuration:
```json
{
  "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
  "quantization": "none",
  "max_model_len": 4096,
  "gpu_memory_utilization": 0.85,
  "enforce_eager": false,
  "hf_token": null
}
```
4. Click **Execute**. The model will download and initialize in VRAM.

### 2. Via cURL from Terminal
```bash
# Load Model
curl -X POST https://<tunnel-id>.trycloudflare.com/admin/load-model \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.85,
    "enforce_eager": False
  }'

# Chat Completion (Streaming)
curl -N -X POST https://<tunnel-id>.trycloudflare.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is quantum entanglement?"}],
    "stream": true,
    "max_tokens": 512
  }'

# Unload Model (Free VRAM back to 0 MB)
curl -X POST https://<tunnel-id>.trycloudflare.com/admin/unload-model
```

### 3. Via the Web UI (`frontend/`)
1. Start the frontend: `cd frontend && npm run dev`.
2. Enter your Cloudflare Quick Tunnel URL in the Settings modal.
3. Use the built-in Model Manager modal to select presets or type custom model IDs.

---

## ⚙️ Environment Configuration Reference (`.env`)

| Variable | Default | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | Server bind host. |
| `PORT` | `8006` | Server bind port. |
| `LOG_LEVEL` | `info` | Uvicorn logging level (`debug`, `info`, `warning`, `error`). |
| `CORS_ORIGINS` | `*` | Allowed CORS origins (comma-delimited or `*`). |
| `MODEL_ID` | `null` | Model ID to load on boot (leave empty for idle start). |
| `AUTO_LOAD_ON_STARTUP` | `false` | If `true` and `MODEL_ID` is set, loads model on startup. |
| `MAX_MODEL_LEN` | `4096` | Max sequence context tokens. |
| `GPU_MEMORY_UTILIZATION` | `0.85` | Fraction of total VRAM reserved for weights and KV cache. |
| `ENFORCE_EAGER` | `true` | Save ~1.5 GB VRAM on T4/L4 by disabling CUDA graph capture. |
| `HF_TOKEN` | `null` | Hugging Face token for gated models (Llama, Gemma). |
| `CLOUDFLARE_TUNNEL_ENABLED`| `true` | Set `false` to disable tunnel and run strictly on localhost. |
| `CLOUDFLARE_TUNNEL_TOKEN` | `null` | Leave empty to automatically use Quick Tunnel (`trycloudflare.com`). |
| `DEFAULT_MAX_TOKENS` | `512` | Default max completion tokens. |
| `DEFAULT_TEMPERATURE` | `0.7` | Sampling temperature (0.0 for greedy). |
| `DEFAULT_ENABLE_THINKING` | `true` | Parse `<think>` tags for reasoning models. |
| `GITHUB_TOKEN` | `null` | GitHub Personal Access Token for private repo Colab cloning. |
| `REPO_URL` | `null` | GitHub repository URL to clone in Colab. |

---

## Validation scope

The catalog's `verified` and `streaming_verified` fields reflect completed isolated-process checks. The validation report records the tested repository, runtime, GPU allocation, inference and SSE outcome, cleanup, and explicit untested candidates. A repository being listed or downloadable does not imply that it has been validated on the current hardware.
