"""Twelve Labs Marengo adapter.

Requires: pip install twelvelabs
Env: TWELVE_LABS_API_KEY

Note: Free tier is rate-limited to 8 req/min. This adapter includes
automatic throttling to stay within limits.
"""

import os
import time
from pathlib import Path

import numpy as np
from twelvelabs import TwelveLabs

from .base import BaseEmbedder, EmbeddingResult

# Twelve Labs free tier: 8 requests per minute
MIN_REQUEST_INTERVAL = 8.0  # seconds between requests (conservative)


class MarengoEmbedder(BaseEmbedder):
    name = "marengo_3"
    display_name = "Twelve Labs Marengo 2.7"
    dimensions = 1024
    is_api = True

    def __init__(self):
        api_key = os.environ.get("TWELVE_LABS_API_KEY")
        if not api_key:
            raise ValueError("TWELVE_LABS_API_KEY environment variable required")
        self.client = TwelveLabs(api_key=api_key)
        self.model_name = "Marengo-retrieval-2.7"
        self._last_request_time = 0.0

    def _throttle(self):
        """Ensure we don't exceed the rate limit."""
        elapsed = time.time() - self._last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def embed_video(self, video_path: Path) -> EmbeddingResult:
        self._throttle()
        start = time.perf_counter()

        # Must pass a file object, not a path string
        with open(video_path, "rb") as f:
            task = self.client.embed.tasks.create(
                model_name=self.model_name,
                video_file=f,
                video_embedding_scope=["clip", "video"],
            )

        # Wait for processing
        self.client.embed.tasks.wait_for_done(task_id=task.id)
        status = self.client.embed.tasks.status(task_id=task.id)

        if status.status != "ready":
            raise RuntimeError(f"Embedding task failed: {status.status}")

        result = self.client.embed.tasks.retrieve(task_id=task.id)
        latency_ms = (time.perf_counter() - start) * 1000

        if result.video_embedding and result.video_embedding.segments:
            vec = np.array(result.video_embedding.segments[0].float_, dtype=np.float32)
        else:
            raise RuntimeError("No video embedding returned")

        return EmbeddingResult(
            vector=self.normalize(vec),
            latency_ms=latency_ms,
            model_name=self.name,
            input_type="video",
        )

    def embed_text(self, text: str) -> EmbeddingResult:
        self._throttle()
        start = time.perf_counter()

        result = self.client.embed.create(
            model_name=self.model_name,
            text=text,
            text_truncate="none",
        )

        latency_ms = (time.perf_counter() - start) * 1000

        if result.text_embedding and result.text_embedding.segments:
            vec = np.array(result.text_embedding.segments[0].float_, dtype=np.float32)
        else:
            err = getattr(result.text_embedding, "error_message", "unknown")
            raise RuntimeError(f"No text embedding returned: {err}")

        return EmbeddingResult(
            vector=self.normalize(vec),
            latency_ms=latency_ms,
            model_name=self.name,
            input_type="text",
        )
