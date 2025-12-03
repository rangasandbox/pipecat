# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Aggregator that buffers the entire LLM response before emitting text."""

from typing import List, Optional

from pipecat.utils.text.base_text_aggregator import Aggregation, AggregationType, BaseTextAggregator


class FullResponseAggregator(BaseTextAggregator):
    """Aggregate the complete response before producing output."""

    def __init__(self):
        self._buffer: List[str] = []

    @property
    def text(self) -> Aggregation:
        return Aggregation(text="".join(self._buffer).strip(), type=AggregationType.SENTENCE)

    async def aggregate(self, text: str) -> Optional[Aggregation]:
        self._buffer.append(text)
        return None

    async def handle_interruption(self):
        self._buffer.clear()

    async def reset(self):
        self._buffer.clear()
