# Resume state: vLLM model catalog task

Updated: 2026-09-26 (America/Los_Angeles)

## Latest continuation (authoritative)

- The user resumed end-to-end API testing across dependency-separated model families. Phi-4 (base runtime), Qwen3.8-27B (vLLM 0.17), Sarvam-30B (vLLM 0.17 + Transformers 5.17), Gemma 4 (vLLM 0.29 CUDA 12.9), and Qwen3-14B (base runtime) all reached READY and returned the requested marker through both normal and SSE APIs. Requests deliberately sent an obsolete model ID; API responses used the active model ID.
- Fixed Gemma 4 on this A100 host (NVIDIA driver R550): the existing vLLM 0.29 environment had a CUDA 13 wheel, which failed in the MoE activation kernel. Replaced that isolated environment's vLLM wheel with the official vLLM 0.29.0 CUDA 12.9 wheel and updated `app/engine.py` to put CUDA 12.9 `nvidia/*/lib` paths first while excluding leftover `nvidia/cu13/lib`. A direct MoE activation check and full Gemma 4 normal/SSE API checks passed. Catalog and README now record Gemma 4 as verified.
- The test round adds Gemma 4 to the previously validated 21 variants: current total is 22/25 verified. The remaining three are Sarvam FP8 (unsupported by A100 SM80), gated Llama 3.3 70B BF16 (three tokens returned 403; BF16 exceeds one A100's VRAM), and gated Aya 32B BF16 (three tokens returned 403).
- `run_colab.sh` was rerun after the runtime change. It stopped the old API on port 8006 and started the runtime-aware app. The currently active model is Qwen3-14B on the base `/usr/bin/python3` runtime; API status is READY. Its normal and SSE responses both returned exact markers. Current tunnel URL: `https://disclaimer-anonymous-responsible-hazardous.trycloudflare.com`.
- Cache management (authorized by the user): evicted Phi-4 and Qwen3.8 caches to make room for Sarvam, then unloaded Sarvam and removed its cache after the successful test, and downloaded/cached Gemma 4. Current disk has about 123 GiB free. Complete caches include the three smaller Llama variants, Qwen3-14B, and Gemma 4. Sarvam, Phi-4, and Qwen3.8 must be downloaded again before another load.
- Verification performed for this continuation: direct API normal/SSE calls, `python3 -m py_compile app/engine.py app/catalog.py`, and restart via `bash run_colab.sh`; no full test suite was run.
- Keep the catalog status and resume counts in this section authoritative if lower sections still contain earlier cache or failure details.

## User objective and current scope

Implement and validate the 15-family, 25-variant model catalog in the inference engine and `frontend-voicebot`, isolate dependency-incompatible runtimes, and record actual end-to-end results. The user authorized deleting cached models, creating separate runtimes, using the supplied Hugging Face credentials for gated checks, and downloading/removing model caches to manage disk space.

The user explicitly resumed testing after asking for a README status update. Current result: 22/25 variants passed; three remain blocked by access or GPU capability, as described in the latest continuation above.

## Workspace and service state

- Main task workspace: `/home/jovyan/frontend-voicebot`.
- Engine code: `/home/jovyan/frontend-voicebot/inference-engine-copy`.
- Active frontend: `/home/jovyan/frontend-voicebot/frontend` (it fetches `/admin/catalog` dynamically).
- Active API on port 8006 runs the runtime-aware engine from `/home/jovyan/vllm-inference-engine-master-copy`; it is idle between requests. Its catalog and backend are synced with the validated implementation.
- Temporary validation API on port 8007 was stopped cleanly.
- Frontend dev server runs on port 5173. `frontend/.env` now points `VITE_VLLM_URL` to the API's current Quick Tunnel URL; it returned HTTP 200 and the Vite module served the updated URL.
- Last observed hardware: NVIDIA A100 80 GB, driver 550.127.08; about 74 GiB allocated to the READY Qwen3-14B engine (vLLM reserves most of the configured 90% GPU budget).
- Last observed disk: about 123 GiB free. Complete model caches: Llama 3.2 3B BF16, Llama 3.2 3B AWQ, Llama 3.1 8B AWQ, Qwen3-14B, and Gemma 4.
- Local ignored `inference-engine-copy/.env` runtime map includes Qwen3.5/Qwen3.5-MoE on vLLM 0.17, Gemma4 on vLLM 0.29, and Sarvam on the new vLLM 0.17 / Transformers 5 runtime.
- An isolated environment exists at `/home/jovyan/.venvs/vllm-0.17.0-tf5`. It has vLLM 0.17.0, PyTorch 2.10.0+cu129, Transformers 5.17.0, Hugging Face Hub 1.33.0, and tokenizers 0.23.2. The existing Qwen runtime remains on Transformers 4.57.6.

## Implemented and checked

- `app/catalog.py` has the authoritative 15-family/25-variant registry; it marks only end-to-end passing variants verified and streaming-verified.
- The frontend reads `/admin/catalog`, uses recommended per-variant parameters, and sends the chosen model ID through `/admin/load-model`.
- Per-family isolated runtime mapping is supported. Sarvam BF16 uses the new Transformers 5 runtime and recommended eager mode; this avoided a Dynamo graph capture failure caused by a Transformers deprecation warning.
- Cache discovery requires non-empty model weights and complete numbered shards. It also handles complete shared Hub 1.x blob references and avoids treating config-only or partial snapshots as cached.
- Model manager checks available disk space before download, runs warm-up inference, proxies non-stream and SSE requests, and terminates model process groups on unload.
- Added `scripts/setup_vllm_017_transformers5_runtime.sh`; `.env.example` and README document the Sarvam runtime.
- Frontend production build passed earlier in this session. Latest `python3 -m compileall -q app`, `bash -n` for runtime setup scripts, and `git diff --check` passed.
- Earlier live `/admin/catalog` check returned 15 families and 25 variants; the current live status has been updated to 22 verified and three remaining candidates.
- Follow-up diagnosis found port 8006 had been serving the original monolithic vLLM 0.15.1 loader. That path could not select the isolated vLLM 0.17 runtime required by Qwen3.8. Port 8006 now serves the validated runtime-aware backend; the original engine checkout was synced so normal restarts retain this routing.
- Fixed a startup error in `run_colab.sh`: it sources `.env`, so the JSON `VLLM_RUNTIME_MAP` value must be wrapped in single quotes to preserve JSON quotes. Updated both local `.env` files, both `.env.example` files, and both fresh-config templates in `run_colab.sh`. Simulated the script's environment sourcing, parsed all four mappings through Pydantic, imported `app.main`, and ran `bash -n` successfully.
- `run_colab.sh` now gracefully stops an existing project Uvicorn process on its configured port before starting. Ran it from the original checkout: it stopped the old API on 8006, dependencies remained at the tested versions, and the new process started from the original checkout. Post-restart Qwen3.8 catalog load selected vLLM 0.17, returned the exact marker in normal/SSE inference, and unloaded cleanly.
- The restart created a new Quick Tunnel URL; `frontend/.env` and the served Vite settings module have been updated to it.
- A LiveKit inference error came from its static `VLLM_MODEL_ID` in `backend/.env` (`Qwen/Qwen2.5-32B-Instruct-AWQ`) while the selected engine model was `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`. The Qwen 2.5 ID is not in the 25-variant registry. Updated `/v1/chat/completions` and `/v1/completions` to always use the active engine model ID, preventing stale client settings from breaking model switches.
- Restarted the original-checkout service with that fix, reloaded the previously selected Llama 3.1 AWQ model, and sent normal and SSE requests carrying the stale Qwen 2.5 model name. Both returned the exact probe marker under the active Llama model; the SSE ended with `[DONE]`. The Llama model remains loaded for voicebot inference.
- Tested Qwen3.8 through port 8006 using its catalog entry: selected the `qwen3.5` vLLM 0.17 runtime, reached READY, returned `MODEL_TEST_OK` in normal and SSE completions, then unloaded to STOPPED. The API's first load request took longer than a short client timeout while CUDA graphs warmed up; it completed successfully.
- The frontend's previous `VITE_VLLM_URL` was stale. Updated it to the current working API tunnel and restarted Vite; the browser-served settings module now contains the matching URL.

## End-to-end variants passed (21)

All of these completed isolated load/warm-up, non-streaming inference, streamed inference ending in `[DONE]`, and shutdown. Reasoning variants used a 160-token cap and were counted only when the requested `MODEL_TEST_OK` marker appeared.

1. `sarvamai/sarvam-30b` — BF16; isolated vLLM 0.17.0 CUDA 12.9 / Transformers 5.17.0, eager mode.
2. `Qwen/Qwen3-30B-A3B` — BF16; base vLLM 0.15.1.
3. `Qwen/Qwen3-30B-A3B-FP8` — FP8 Marlin on A100; vLLM 0.15.1.
4. `Qwen/Qwen3-14B` — BF16; vLLM 0.15.1.
5. `casperhansen/llama-3.3-70b-instruct-awq` — AWQ; vLLM 0.15.1.
6. `meta-llama/Llama-3.2-3B-Instruct` — BF16; gated download with supplied token.
7. `casperhansen/llama-3.2-3b-instruct-awq` — AWQ; vLLM 0.15.1.
8. `meta-llama/Llama-3.1-8B-Instruct` — BF16; gated download with credential 2 or 3.
9. `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` — AWQ; vLLM 0.15.1.
10. `google/gemma-3-27b-it` — BF16; gated download with supplied token.
11. `microsoft/phi-4` — BF16; vLLM 0.15.1.
12. `Qwen/Qwen3.6-27B` — BF16; isolated vLLM 0.17.
13. `Qwen/Qwen3.6-27B-FP8` — FP8 Marlin; isolated vLLM 0.17.
14. `Qwen/Qwen3.6-35B-A3B` — BF16; isolated vLLM 0.17; about 75 GiB peak model memory.
15. `Qwen/Qwen3.6-35B-A3B-FP8` — FP8; isolated vLLM 0.17.
16. `Qwen/Qwen3.8-27B` — BF16; isolated vLLM 0.17.
17. `Qwen/Qwen3.8-27B-FP8` — FP8 Marlin; isolated vLLM 0.17.
18. `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` — BF16; vLLM 0.15.1.
19. `casperhansen/deepseek-r1-distill-qwen-32b-awq` — AWQ; vLLM 0.15.1.
20. `mistralai/Mistral-Small-3.2-24B-Instruct-2506` — BF16; vLLM 0.15.1.
21. `Orion-zhen/aya-expanse-32b-AWQ` — AWQ; vLLM 0.15.1.

Additionally, a verified UI-compatible selection path was exercised using `casperhansen/llama-3.2-3b-instruct-awq`: catalog selection/load reached READY, non-stream and SSE both returned the exact marker, and unload completed.

## Three remaining variants and blockers

- `sarvamai/sarvam-30b-fp8`: vLLM ModelOpt FP8 requires compute capability 8.9; current A100 is SM80. Marked unsupported on this GPU.
- `meta-llama/Llama-3.3-70B-Instruct` BF16: all three supplied credentials returned HTTP 403 on file access. The roughly 140 GB BF16 weights also exceed this single 80 GB GPU.
- `CohereLabs/aya-expanse-32b` BF16: all three supplied credentials returned HTTP 403 on config/file access.

Credential access checks: supplied credentials had file access for Llama 3.2 and Gemma 3; credentials 2/3 had file access for Llama 3.1. No token values are saved in workspace reports.

## Current documentation and report

- `README.md` now describes isolated runtime routing and the CUDA 12.9 Gemma 4 fix. The prior validation report still lists the earlier 21/25 state and needs a follow-up update if maintained as the final report.
- `model-validation-report.md` contains one row for all 25 variants with runtime, streaming, shutdown, cache state, and precise failure/access reasons.

## Next action if the user resumes

Only the three access/hardware blockers above remain. A future run can address Sarvam FP8 on SM89+, Llama 3.3 70B BF16 with repo access and adequate multi-GPU VRAM, or Aya BF16 with repo access.

Before final status, leave port 8006 and Vite on port 5173 running, preserve unrelated dirty/staged outer-repository changes, and report that four catalog candidates remain blocked as documented above.
