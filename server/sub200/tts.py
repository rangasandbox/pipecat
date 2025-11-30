"""FastAPI proxy for the Sub200 TTS HTTP endpoint.

Run with::

    uvicorn server.sub200.tts:app --reload

The server mirrors the `/v1/tts/generate` API used in the provided curl
snippet, which makes it easy to test the Sub200 integration locally or from
automation tools that expect an HTTP endpoint.
"""

from __future__ import annotations

import os
from typing import AsyncIterator

import aiohttp
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


def _str_to_bool(value: str, default: bool = True) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


DEFAULT_BASE_URL = os.getenv("SUB200_TTS_BASE_URL", "http://136.116.73.7/v1/tts/generate")
DEFAULT_DESCRIPTION = os.getenv(
    "SUB200_TTS_DESCRIPTION", "Male voice in their 30s with american accent"
)
DEFAULT_STREAM = _str_to_bool(os.getenv("SUB200_TTS_STREAM", "true"))
DEFAULT_TIMEOUT = float(os.getenv("SUB200_TTS_TIMEOUT", "30"))

app = FastAPI(title="Sub200 TTS proxy")


class TTSRequest(BaseModel):
    text: str
    description: str | None = None
    stream: bool | None = None


async def _forward_request(payload: dict) -> AsyncIterator[bytes]:
    timeout = aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT) if DEFAULT_TIMEOUT else None
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            DEFAULT_BASE_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as response:
            if response.status != 200:
                detail = await response.text()
                raise HTTPException(status_code=response.status, detail=detail)
            async for chunk in response.content.iter_chunked(8192):
                yield chunk


@app.post("/v1/tts/generate")
async def generate_tts(request: TTSRequest):
    payload = {
        "text": request.text,
        "description": request.description or DEFAULT_DESCRIPTION,
        "stream": DEFAULT_STREAM if request.stream is None else request.stream,
    }

    return StreamingResponse(
        _forward_request(payload),
        media_type="audio/wav",
        headers={"Content-Disposition": "inline; filename=tts.wav"},
    )


if __name__ == "__main__":  # pragma: no cover - dev server helper
    import uvicorn

    uvicorn.run("server.sub200.tts:app", host="0.0.0.0", port=8081, reload=False)
