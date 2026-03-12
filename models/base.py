"""Base class for embedding model adapters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class EmbeddingResult:
    """Result from embedding a single input."""

    vector: np.ndarray  # normalized to unit length
    latency_ms: float  # wall-clock time to compute embedding
    model_name: str
    input_type: str  # "video" or "text"


class BaseEmbedder(ABC):
    """Interface that all model adapters implement."""

    name: str  # e.g. "gemini_embedding_2"
    display_name: str  # e.g. "Gemini Embedding 2"
    dimensions: int
    is_api: bool  # True for cloud APIs, False for local models

    @abstractmethod
    def embed_video(self, video_path: Path) -> EmbeddingResult:
        """Embed a video file. Returns a single vector."""
        ...

    @abstractmethod
    def embed_text(self, text: str) -> EmbeddingResult:
        """Embed a text query. Returns a single vector."""
        ...

    def embed_videos_batch(self, video_paths: list[Path]) -> list[EmbeddingResult]:
        """Embed multiple videos. Default: sequential. Override for batching."""
        return [self.embed_video(p) for p in video_paths]

    def embed_texts_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed multiple texts. Default: sequential. Override for batching."""
        return [self.embed_text(t) for t in texts]

    @staticmethod
    def normalize(v: np.ndarray) -> np.ndarray:
        """L2-normalize a vector."""
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm
