#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os
import pathlib
import sys

# Allow running the script without installing the package.
PROJECT_SRC = pathlib.Path(__file__).resolve().parent / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import Frame, LLMRunFrame, MetricsFrame
from pipecat.metrics.metrics import (
    LLMUsageMetricsData,
    ProcessingMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from deepgram import LiveOptions

from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.sub200.tts import Sub200TTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.transcriptions.language import Language

load_dotenv(override=True)


class MetricsLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, MetricsFrame):
            for d in frame.data:
                if isinstance(d, TTFBMetricsData):
                    print(f"!!! MetricsFrame: {frame}, ttfb: {d.value}")
                elif isinstance(d, ProcessingMetricsData):
                    print(f"!!! MetricsFrame: {frame}, processing: {d.value}")
                elif isinstance(d, LLMUsageMetricsData):
                    tokens = d.value
                    print(
                        f"!!! MetricsFrame: {frame}, tokens: {tokens.prompt_tokens}, characters: {tokens.completion_tokens}"
                    )
                elif isinstance(d, TTSUsageMetricsData):
                    print(f"!!! MetricsFrame: {frame}, characters: {d.value}")
        await self.push_frame(frame, direction)


# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
def _daily_transport_params():
    try:
        from pipecat.transports.daily.transport import DailyParams
    except Exception as exc:  # pragma: no cover - import-time dependency guard
        raise RuntimeError(
            "Daily transport requires the 'pipecat-ai[daily]' extra: pip install pipecat-ai[daily]"
        ) from exc

    return DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    )


transport_params = {
    "daily": _daily_transport_params,
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        live_options=LiveOptions(language=Language.HI, model="nova-3-general"),
    )

    sub200_base_url = os.getenv(
        "SUB200_TTS_BASE_URL", "http://tts.sub200.dev/indic-19/v1/tts/generate"
    )
    sub200_voice = os.getenv(
        "SUB200_TTS_VOICE", os.getenv("SUB200_TTS_VOICE_ID", "Shashank")
    )
    sub200_stream = os.getenv("SUB200_TTS_STREAM", "true").lower() in ("1", "true", "yes")
    sub200_timeout = float(os.getenv("SUB200_TTS_TIMEOUT", "30"))
    sub200_sample_rate = os.getenv("SUB200_TTS_SAMPLE_RATE")

    tts = Sub200TTSService(
        base_url=sub200_base_url,
        voice=sub200_voice,
        stream=sub200_stream,
        timeout=sub200_timeout,
        sample_rate=int(sub200_sample_rate) if sub200_sample_rate else None,
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    ml = MetricsLogger()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly assistant speaking with a user over a WebRTC call. "
                "Always respond entirely in Hindi, keep the answers concise, and avoid using "
                "special characters or lists that would be awkward to pronounce."
            ),
        },
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            ml,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
