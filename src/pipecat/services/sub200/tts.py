"""Sub200 text-to-speech service implementation.

This module integrates Pipecat with Sub200's HTTP TTS endpoint. The service
streams WAV audio returned by the Sub200 API, strips the WAV header, and
emits raw PCM frames compatible with the rest of the Pipecat audio pipeline.
"""

from __future__ import annotations

from typing import AsyncGenerator, Optional

import aiohttp
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TTSStartedFrame, TTSStoppedFrame
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts


class Sub200TTSService(TTSService):
    """HTTP-based Sub200 TTS service.

    Sends POST requests to the Sub200 `/v1/tts/generate` endpoint and streams
    back PCM audio frames.
    """

    def __init__(
        self,
        *,
        base_url: str,
        voice: str,
        stream: bool = True,
        temperature: float = 0.0,
        timeout: Optional[float] = 30.0,
        sample_rate: Optional[int] = None,
        **kwargs,
    ):
        """Initialize the Sub200 TTS service.

        Args:
            base_url: Full URL of the Sub200 TTS endpoint.
            voice: Voice identifier provided by Sub200 (e.g. "Shashank").
            stream: Whether the Sub200 endpoint should stream output audio.
            temperature: Sampling temperature for Sub200 synthesis (0-1).
            timeout: Optional request timeout in seconds.
            sample_rate: Preferred sample rate. If None, Pipecat's start frame
                sample rate is used until updated dynamically from the WAV
                header.
            **kwargs: Additional arguments passed to :class:`TTSService`.
        """

        super().__init__(sample_rate=sample_rate, **kwargs)
        self._base_url = base_url.rstrip("/")
        self._voice = voice
        self._stream = stream
        self._temperature = temperature
        self._timeout = timeout

    def can_generate_metrics(self) -> bool:
        """Whether the service can emit processing metrics."""

        return True

    async def _iter_response_chunks(self, response: aiohttp.ClientResponse):
        chunk_iterator = response.content.iter_chunked(max(self.chunk_size, 4096))

        try:
            first_chunk = await chunk_iterator.__anext__()
        except StopAsyncIteration:
            return

        if first_chunk.startswith(b"RIFF") and len(first_chunk) >= 28:
            sample_rate = int.from_bytes(first_chunk[24:28], "little")
            if sample_rate and sample_rate != self.sample_rate:
                logger.debug(f"{self}: switching sample rate to {sample_rate}Hz from WAV header")
                self._sample_rate = sample_rate

        async def chunk_generator():
            yield first_chunk
            async for chunk in chunk_iterator:
                yield chunk

        async for chunk in chunk_generator():
            yield chunk

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate audio by calling the Sub200 HTTP endpoint."""

        print(f"{self}: Generating TTS [{text}]")
        logger.info(
            f"{self}: Invoking Sub200 TTS at {self._base_url} with voice={self._voice}, "
            f"temperature={self._temperature}, stream={self._stream}"
        )
        print(
            f"[Sub200TTS] text={text} voice={self._voice} temp={self._temperature} stream={self._stream}"
        )
        payload = {
            "voice": self._voice,
            "text": text,
            "stream": self._stream,
            "temperature": self._temperature,
        }

        timeout = aiohttp.ClientTimeout(total=self._timeout) if self._timeout else None

        await self.start_ttfb_metrics()

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self._base_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status != 200:
                        error_body = await response.text()
                        await self.stop_ttfb_metrics()
                        yield ErrorFrame(
                            error=(
                                f"Error getting audio (status: {response.status}, error: {error_body})"
                            )
                        )
                        return

                    await self.start_tts_usage_metrics(text)
                    yield TTSStartedFrame()

                    measuring_ttfb = True
                    chunk_iterator = self._iter_response_chunks(response)

                    async for frame in self._stream_audio_frames_from_iterator(
                        chunk_iterator, strip_wav_header=True
                    ):
                        if measuring_ttfb:
                            await self.stop_ttfb_metrics()
                            measuring_ttfb = False
                        yield frame

                    if measuring_ttfb:
                        await self.stop_ttfb_metrics()

        except Exception as exc:  # pragma: no cover - network failures
            logger.error(f"{self} exception: {exc}")
            await self.stop_ttfb_metrics()
            yield ErrorFrame(error=f"Unknown error occurred: {exc}")
        finally:
            yield TTSStoppedFrame()
