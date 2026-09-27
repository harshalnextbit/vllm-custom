#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_dir="${1:-${repo_root}/.runtimes/vllm-0.17.0}"

python3 -m venv "${runtime_dir}"
"${runtime_dir}/bin/pip" install \
  'vllm==0.17.0' \
  'transformers==4.57.6' \
  --extra-index-url https://download.pytorch.org/whl/cu129

printf '\nAdd this to the inference engine .env file for Qwen3.6-family models:\n'
printf 'VLLM_RUNTIME_MAP={"qwen3.5":"%s/bin/python","qwen3.5-moe":"%s/bin/python"}\n' \
  "${runtime_dir}" "${runtime_dir}"
