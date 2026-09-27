# vLLM model catalog validation report

Validation date: 2026-09-26. Scope: all 25 catalog variants on NVIDIA A100 80 GB, host driver 550.127.08. Base runtime: vLLM 0.15.1 / Transformers 4.57.6. Qwen3.6/Qwen3.8 runtime: isolated vLLM 0.17.0 with CUDA 12.9 wheels. Sarvam runtime: isolated vLLM 0.17.0 / Transformers 5.17.0 with eager execution. Gemma 4 runtime: isolated vLLM 0.29.0 CUDA 12.9 wheel / Transformers 5.17.0.

After identifying that the pre-existing API process used the old monolithic vLLM 0.15.1 loader, port 8006 was switched to the runtime-aware backend. Qwen3.8-27B was loaded via its catalog selection on port 8006, which routed it to vLLM 0.17.0; normal and streamed marker inference passed. A later API round also passed Phi-4, Qwen3.8-27B, Sarvam-30B, Gemma 4, and Qwen3-14B normal/SSE probes with intentionally stale client model IDs. Gemma 4 initially failed with the CUDA 13.0 wheel on the R550 host; replacing it with the official vLLM 0.29.0 CUDA 12.9 wheel fixed the MoE warm-up and both API response modes.

22 of 25 variants passed isolated model load, warm-up, non-streaming inference, streaming inference through `[DONE]`, and clean shutdown. For reasoning models, the completion probe allowed 160 tokens and required the requested marker to appear. Three remain unverified or unsupported for the exact reasons below.

| Model | Hugging Face ID | Variant | Architecture | Load / warm-up | Nonstream | SSE | Shutdown | Weights cached now | Result |
|---|---|---|---|---|---|---|---|---:|---|
| Sarvam-30B | `sarvamai/sarvam-30b` | BF16 / bfloat16 | — | PASS | PASS | PASS | PASS | No | PASS — isolated vLLM 0.17.0 CUDA 12.9 / Transformers 5.17.0 with eager mode; marker generated and stream ended with [DONE]. |
| Sarvam-30B | `sarvamai/sarvam-30b-fp8` | FP8 / fp8 | — | NOT RUN | NOT RUN | NOT RUN | N/A | No | UNSUPPORTED — vLLM ModelOpt FP8 requires compute capability 8.9; current A100 is SM80 (capability 8.0). |
| Qwen3-30B-A3B | `Qwen/Qwen3-30B-A3B` | BF16 / bfloat16 | — | PASS | PASS | PASS | PASS | No | PASS — marker generated; stream ended with [DONE]. |
| Qwen3-30B-A3B | `Qwen/Qwen3-30B-A3B-FP8` | FP8 / fp8 | — | PASS | PASS | PASS | PASS | No | PASS — A100 used weight-only FP8 via Marlin; marker generated and stream ended with [DONE]. |
| Qwen3-14B | `Qwen/Qwen3-14B` | BF16 / bfloat16 | — | PASS | PASS | PASS | PASS | Yes | PASS — marker generated; stream ended with [DONE]. |
| Llama 3.3 70B | `meta-llama/Llama-3.3-70B-Instruct` | BF16 (gated) / bfloat16 | — | NOT RUN | NOT RUN | NOT RUN | N/A | No | NOT RUN — all three supplied credentials received HTTP 403 for the gated repo. BF16 checkpoint is about 140 GB, beyond the 80 GB GPU. |
| Llama 3.3 70B | `casperhansen/llama-3.3-70b-instruct-awq` | AWQ / awq | — | PASS | PASS | PASS | PASS | No | PASS — marker generated; stream ended with [DONE]. |
| Llama 3.2 3B | `meta-llama/Llama-3.2-3B-Instruct` | BF16 (gated) / bfloat16 | — | PASS | PASS | PASS | PASS | Yes | PASS — gated download with authorized supplied token; marker generated, stream ended with [DONE]. |
| Llama 3.2 3B | `casperhansen/llama-3.2-3b-instruct-awq` | AWQ / awq | — | PASS | PASS | PASS | PASS | Yes | PASS — marker generated; stream ended with [DONE]. |
| Llama 3.1 8B Instruct | `meta-llama/Llama-3.1-8B-Instruct` | BF16 (gated) / bfloat16 | — | PASS | PASS | PASS | PASS | No | PASS — gated download with authorized supplied token; marker generated, stream ended with [DONE]. |
| Llama 3.1 8B Instruct | `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` | AWQ / awq | — | PASS | PASS | PASS | PASS | Yes | PASS — marker generated; stream ended with [DONE]. |
| Gemma 3 27B IT | `google/gemma-3-27b-it` | BF16 (gated) / bfloat16 | — | PASS | PASS | PASS | PASS | No | PASS — gated download with authorized supplied token; marker generated, stream ended with [DONE]. |
| Phi-4 14B | `microsoft/phi-4` | BF16 / bfloat16 | — | PASS | PASS | PASS | PASS | No | PASS — marker generated; stream ended with [DONE]. Cache was removed after the API test to free disk space. |
| Qwen3.6-27B | `Qwen/Qwen3.6-27B` | BF16 / bfloat16 | — | PASS | PASS | PASS | PASS | No | PASS — marker generated; stream ended with [DONE]. |
| Qwen3.6-27B | `Qwen/Qwen3.6-27B-FP8` | FP8 / fp8 | — | PASS | PASS | PASS | PASS | No | PASS — marker generated; stream ended with [DONE]. |
| Qwen3.6-35B-A3B | `Qwen/Qwen3.6-35B-A3B` | BF16 / bfloat16 | — | PASS | PASS | PASS | PASS | No | PASS — vLLM 0.17.0; model used about 75 GiB VRAM; marker generated and stream ended with [DONE]. |
| Qwen3.6-35B-A3B | `Qwen/Qwen3.6-35B-A3B-FP8` | FP8 / fp8 | — | PASS | PASS | PASS | PASS | No | PASS — marker generated; stream ended with [DONE]. |
| Qwen3.8-27B | `Qwen/Qwen3.8-27B` | BF16 / bfloat16 | — | PASS | PASS | PASS | PASS | No | PASS — marker generated; stream ended with [DONE]; active API selected the isolated vLLM 0.17.0 runtime. Cache was removed after the API test to free disk space. |
| Qwen3.8-27B | `Qwen/Qwen3.8-27B-FP8` | FP8 / fp8 | — | PASS | PASS | PASS | PASS | No | PASS — A100 used weight-only FP8 via Marlin; marker generated and stream ended with [DONE]. |
| Gemma 4 26B-A4B-IT | `google/gemma-4-26B-A4B-it` | BF16 / bfloat16 | — | PASS | PASS | PASS | PASS | Yes | PASS — vLLM 0.29.0 CUDA 12.9 wheel avoids the R550/CUDA 13 driver failure; loaded 48.54 GiB and passed warm-up plus normal/SSE marker inference. |
| DeepSeek-R1-Distill-Qwen-32B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | BF16 / bfloat16 | — | PASS | PASS | PASS | PASS | No | PASS — reasoning output included MODEL_TEST_OK with max_tokens=160; stream ended with [DONE]. |
| DeepSeek-R1-Distill-Qwen-32B | `casperhansen/deepseek-r1-distill-qwen-32b-awq` | AWQ / awq | — | PASS | PASS | PASS | PASS | No | PASS — reasoning output included MODEL_TEST_OK with max_tokens=160; stream ended with [DONE]. |
| Mistral Small 3.2 24B | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | BF16 / bfloat16 | — | PASS | PASS | PASS | PASS | No | PASS — model use 44.76 GiB; marker generated and stream ended with [DONE]. |
| Aya Expanse 32B | `CohereLabs/aya-expanse-32b` | BF16 (gated) / bfloat16 | — | NOT RUN | NOT RUN | NOT RUN | N/A | No | NOT RUN — all three supplied credentials received HTTP 403 for this gated repository. |
| Aya Expanse 32B | `Orion-zhen/aya-expanse-32b-AWQ` | AWQ / awq | — | PASS | PASS | PASS | PASS | No | PASS — marker generated; stream ended with [DONE]. |

## Catalog selection and inference

Fetched `/admin/catalog`, selected the verified `casperhansen/llama-3.2-3b-instruct-awq` entry by its Hugging Face ID, and loaded it through the frontend-compatible `/admin/load-model` payload. The engine reached READY; normal and SSE inference returned `MODEL_TEST_OK`, SSE ended with `[DONE]`, and unload returned STOPPED. Sarvam BF16 was separately loaded through its catalog runtime with eager mode enabled and passed normal, streamed, and shutdown checks.

Gated access checks used the three credentials supplied for this task. File access was available for Llama 3.2, Llama 3.1 using credentials 2/3, and Gemma 3. The supplied credentials returned HTTP 403 for Llama 3.3 70B and Aya Expanse base. Tokens were not written into this report.

## Hardware and compatibility constraints

- Sarvam-30B BF16 is routed to the isolated Transformers 5 runtime; vLLM 0.17 with Transformers 4 cannot load its MoE architecture.
- Sarvam-30B FP8: ModelOpt FP8 requires SM89; this A100 is SM80.
- Gemma 4 must use the vLLM 0.29.0 CUDA 12.9 wheel on this R550 host; its CUDA 13.0 wheel fails the MoE activation kernel. The CUDA 12.9 wheel passed warm-up and API inference.
- Llama 3.3 70B BF16: gated access denied for the supplied credentials; its BF16 weights also exceed this single A100’s VRAM.
- Aya Expanse 32B BF16: gated access denied for the supplied credentials.

## Cache and cleanup

Cache discovery requires non-empty model-weight files and checks that numbered shards are complete. It also recognizes complete shared Hugging Face blob sets from Hub 1.x `.refs` metadata. Config-only snapshots and partial weight-shard sets are not treated as cached. Test workers were switched cleanly; Qwen3-14B is currently loaded on port 8006 and Gemma 4, Qwen3-14B, and three small Llama variants remain cached. Sarvam-30B, Phi-4, and Qwen3.8 caches were removed after testing to preserve disk space.
