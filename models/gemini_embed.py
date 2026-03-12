"""Google Gemini Embedding 2 adapter.

Requires: pip install google-genai
Env: GOOGLE_API_KEY
"""

import os
import time
from pathlib import Path

import numpy as np
from google import genai
from google.genai import types

from .base import BaseEmbedder, EmbeddingResult


class GeminiEmbedder(BaseEmbedder):
    name = "gemini_embedding_2"
    display_name = "Gemini Embedding 2"
    dimensions = 3072
    is_api = True

    def __init__(self, task_type: str = "RETRIEVAL_DOCUMENT", dimensions: int = 3072):
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable required")
        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-embedding-2-preview"  # Embedding 2 model ID
        self.task_type = task_type
        self.dimensions = dimensions

    def embed_video(self, video_path: Path) -> EmbeddingResult:
        video_file = self.client.files.upload(file=video_path)

        # Wait for processing
        while video_file.state.name == "PROCESSING":
            time.sleep(1)
            video_file = self.client.files.get(name=video_file.name)

        if video_file.state.name != "ACTIVE":
            raise RuntimeError(f"Video upload failed: {video_file.state.name}")

        start = time.perf_counter()
        response = self.client.models.embed_content(
            model=self.model_id,
            contents=video_file,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=self.dimensions,
            ),
        )
        latency_ms = (time.perf_counter() - start) * 1000

        vec = np.array(response.embeddings[0].values, dtype=np.float32)

        # Cleanup uploaded file
        try:
            self.client.files.delete(name=video_file.name)
        except Exception:
            pass

        return EmbeddingResult(
            vector=self.normalize(vec),
            latency_ms=latency_ms,
            model_name=self.name,
            input_type="video",
        )

    def embed_text(self, text: str) -> EmbeddingResult:
        start = time.perf_counter()
        response = self.client.models.embed_content(
            model=self.model_id,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=self.dimensions,
            ),
        )
        latency_ms = (time.perf_counter() - start) * 1000

        vec = np.array(response.embeddings[0].values, dtype=np.float32)
        return EmbeddingResult(
            vector=self.normalize(vec),
            latency_ms=latency_ms,
            model_name=self.name,
            input_type="text",
        )
