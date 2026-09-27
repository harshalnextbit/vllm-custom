import json

import httpx
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse

from app.engine import get_vllm_manager
from app.schemas.requests import CompletionRequest

router = APIRouter(tags=["Inference"])


@router.post("/v1/completions")
async def completions(req: CompletionRequest):
    vm = get_vllm_manager()
    if not vm.is_loaded() or not vm.api_base_url:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="No model is currently ready.", headers={"Retry-After": "5"})
    payload = req.model_dump(exclude_none=True)
    # Keep the selected backend model authoritative when a client submits a
    # stale model name from its own startup configuration.
    payload["model"] = vm.model_id
    payload.pop("do_sample", None)
    try:
        if not req.stream:
            return JSONResponse(content=await vm.proxy_request("completions", payload))
        url = f"{vm.api_base_url}/completions"

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
