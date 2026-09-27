#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_dir="${1:-${repo_root}/.runtimes/vllm-0.17.0-transformers5}"

python3 -m venv "${runtime_dir}"
"${runtime_dir}/bin/pip" install \
  'vllm==0.17.0' \
  --extra-index-url https://download.pytorch.org/whl/cu129

# vLLM 0.17 declares transformers<5, but Sarvam's Transformers modeling
# backend requires Transformers 5. Keep this override in a separate runtime;
# eager execution avoids a Transformers 5 deprecation warning inside Dynamo.
"${runtime_dir}/bin/pip" install --no-deps \
  'transformers==5.17.0' \
  'huggingface-hub==1.33.0' \
  'tokenizers==0.23.2'

printf '\nAdd this entry to VLLM_RUNTIME_MAP for Sarvam:\n'
printf '"sarvam":"%s/bin/python"\n' "${runtime_dir}"
