import json

import httpx
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import get_settings
from app.engine import get_vllm_manager
from app.schemas.requests import ChatCompletionRequest

router = APIRouter(tags=["Inference"])


@router.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    vm = get_vllm_manager()
    if not vm.is_loaded() or not vm.api_base_url:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="No model is currently ready.", headers={"Retry-After": "5"})

    payload = req.model_dump(exclude_none=True)
    # This engine hosts one selected model at a time. Clients such as LiveKit
    # keep their model name from startup config, which can become stale after a
    # model switch; always route to the currently loaded model.
    payload["model"] = vm.model_id
    # These are application extensions; vLLM's OpenAI endpoint only receives
    # standard completion parameters. Its model chat template controls reasoning.
    requested_thinking = req.resolved_enable_thinking()
    enable_thinking = get_settings().DEFAULT_ENABLE_THINKING if requested_thinking is None else requested_thinking
    model_id = (vm.model_id or "").casefold()
    if "sarvam" in model_id:
        enable_thinking = False
    if any(name in model_id for name in ("qwen", "deepseek", "sarvam")):
        payload.setdefault("chat_template_kwargs", {})["enable_thinking"] = enable_thinking
        # Sarvam's bundled chat template supports enable_thinking and adds the
        # <|nothink|> marker to the user turn when it is disabled. Qwen and
        # DeepSeek use the local override below for their non-thinking format.
        if not enable_thinking and ("qwen" in model_id or "deepseek" in model_id):
            with open("/home/jovyan/template.txt", "r") as f:
                payload["chat_template"] = f.read().strip()
    if req.max_tokens is None:
        payload["max_tokens"] = get_settings().DEFAULT_MAX_TOKENS
    for key in ("enable_thinking", "thinking", "thinking_budget", "do_sample"):
        payload.pop(key, None)
    try:
        if not req.stream:
            result = await vm.proxy_request("chat/completions", payload)
            return JSONResponse(content=result)

        url = f"{vm.api_base_url}/chat/completions"

        async def stream():
            try:
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream("POST", url, json=payload) as response:
                        if response.is_error:
                            body = await response.aread()
                            yield f"data: {json.dumps({'error': {'message': body.decode('utf-8', 'replace'), 'type': 'upstream_error'}})}\n\n"
                            yield "data: [DONE]\n\n"
                            return
                        async for chunk in response.aiter_raw():
                            if chunk:
                                yield chunk
            except Exception as exc:
                yield f"data: {json.dumps({'error': {'message': str(exc), 'type': type(exc).__name__}})}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
