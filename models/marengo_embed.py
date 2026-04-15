"""Twelve Labs Marengo adapter (Embed API v2).

Requires: pip install twelvelabs
Env: TWELVE_LABS_API_KEY

Note: Free tier is rate-limited to 8 req/min. This adapter includes
automatic throttling to stay within limits.

v2 flow:
  1. Upload the local video as an Asset.
  2. Create an async embed task referencing the asset_id.
  3. Poll the task until status == "ready".
  4. Read the (single) asset-scope embedding from the response.
"""

import os
import time
from pathlib import Path

import numpy as np
from twelvelabs import MediaSource, TextInputRequest, TwelveLabs, VideoInputRequest

from .base import BaseEmbedder, EmbeddingResult

# Twelve Labs free tier: 8 requests per minute
MIN_REQUEST_INTERVAL = 8.0  # seconds between requests (conservative)

# Async embed-task polling
TASK_POLL_INTERVAL = 2.0   # seconds between status checks
TASK_POLL_TIMEOUT = 600.0  # give up after 10 minutes

# Asset upload polling (direct uploads are usually synchronous, but handle
# the "processing" state in case the server defers validation).
ASSET_POLL_INTERVAL = 1.0
ASSET_POLL_TIMEOUT = 120.0


class MarengoEmbedder(BaseEmbedder):
    name = "marengo_3"
    display_name = "Twelve Labs Marengo 3.0"
    dimensions = 1024
    is_api = True

    def __init__(self):
        api_key = os.environ.get("TWELVE_LABS_API_KEY")
        if not api_key:
            raise ValueError("TWELVE_LABS_API_KEY environment variable required")
        self.client = TwelveLabs(api_key=api_key)
        self.model_name = "marengo3.0"
        self._last_request_time = 0.0

    def _throttle(self):
        """Ensure we don't exceed the rate limit."""
        elapsed = time.time() - self._last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def _wait_for_asset(self, asset_id: str):
        """Block until an uploaded asset is ready (or fail fast)."""
        deadline = time.time() + ASSET_POLL_TIMEOUT
        while True:
            asset = self.client.assets.retrieve(asset_id=asset_id)
            if asset.status == "ready" or asset.status is None:
                return asset
            if asset.status == "failed":
                raise RuntimeError(f"Asset {asset_id} failed to process")
            if time.time() > deadline:
                raise RuntimeError(
                    f"Asset {asset_id} not ready after {ASSET_POLL_TIMEOUT}s"
                )
            time.sleep(ASSET_POLL_INTERVAL)

    def _wait_for_task(self, task_id: str):
        """Poll an embed task until it reaches a terminal state."""
        deadline = time.time() + TASK_POLL_TIMEOUT
        while True:
            result = self.client.embed.v_2.tasks.retrieve(task_id=task_id)
            if result.status == "ready":
                return result
            if result.status == "failed":
                raise RuntimeError(f"Embed task {task_id} failed")
            if time.time() > deadline:
                raise RuntimeError(
                    f"Embed task {task_id} timed out after {TASK_POLL_TIMEOUT}s"
                )
            time.sleep(TASK_POLL_INTERVAL)

    def embed_video(self, video_path: Path) -> EmbeddingResult:
        self._throttle()
        start = time.perf_counter()

        # 1. Upload the video as an asset
        with open(video_path, "rb") as f:
            asset = self.client.assets.create(method="direct", file=f)
        if not asset.id:
            raise RuntimeError("Asset upload returned no id")
        self._wait_for_asset(asset.id)

        # 2. Kick off the async embed task (asset-level embedding = one
        #    1024-d vector for the whole video)
        task = self.client.embed.v_2.tasks.create(
            input_type="video",
            model_name=self.model_name,
            video=VideoInputRequest(
                media_source=MediaSource(asset_id=asset.id),
                embedding_option=["visual"],
                embedding_scope=["asset"],
            ),
        )

        # 3. Poll until done
        result = self._wait_for_task(task.id)
        latency_ms = (time.perf_counter() - start) * 1000

        if not result.data:
            raise RuntimeError("No video embedding returned")

        vec = np.array(result.data[0].embedding, dtype=np.float32)

        return EmbeddingResult(
            vector=self.normalize(vec),
            latency_ms=latency_ms,
            model_name=self.name,
            input_type="video",
        )

    def embed_text(self, text: str) -> EmbeddingResult:
        self._throttle()
        start = time.perf_counter()

        result = self.client.embed.v_2.create(
            input_type="text",
            model_name=self.model_name,
            text=TextInputRequest(input_text=text),
        )

        latency_ms = (time.perf_counter() - start) * 1000

        if not result.data:
            raise RuntimeError("No text embedding returned")

        vec = np.array(result.data[0].embedding, dtype=np.float32)

        return EmbeddingResult(
            vector=self.normalize(vec),
            latency_ms=latency_ms,
            model_name=self.name,
            input_type="text",
        )
