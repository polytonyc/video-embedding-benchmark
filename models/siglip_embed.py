"""SigLIP 2 SO400M adapter (local, open-source, frame-based).

This is an image-text model — no native video understanding. We embed videos
by sampling N frames and mean-pooling the frame embeddings. This serves as a
baseline to show the gap between frame-sampling and temporal video models.

Requires: pip install transformers torch Pillow
GPU recommended, works on CPU/MPS.
"""

import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from .base import BaseEmbedder, EmbeddingResult

try:
    from decord import VideoReader, cpu as decord_cpu
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False
    import cv2


def extract_pil_frames(video_path: Path, num_frames: int = 8) -> list[Image.Image]:
    """Extract evenly-spaced frames as PIL Images."""
    if HAS_DECORD:
        vr = VideoReader(str(video_path), ctx=decord_cpu(0))
        total = len(vr)
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()
        return [Image.fromarray(f) for f in frames]
    else:
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        pil_frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                pil_frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        return pil_frames


class SigLIPEmbedder(BaseEmbedder):
    name = "siglip2_so400m"
    display_name = "SigLIP 2 SO400M (frame-avg)"
    dimensions = 1152
    is_api = False

    def __init__(self, num_frames: int = 8):
        self.num_frames = num_frames
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        model_name = "google/siglip2-so400m-patch14-384"
        print(f"Loading SigLIP 2 SO400M on {self.device}...")

        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)

    def _embed_images(self, images: list[Image.Image]) -> np.ndarray:
        """Embed a batch of PIL images. Returns (N, dim) array."""
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        return outputs.cpu().numpy().astype(np.float32)

    def embed_video(self, video_path: Path) -> EmbeddingResult:
        start = time.perf_counter()

        frames = extract_pil_frames(video_path, self.num_frames)
        frame_embeddings = self._embed_images(frames)

        # Mean-pool frame embeddings into a single video vector
        vec = frame_embeddings.mean(axis=0)

        latency_ms = (time.perf_counter() - start) * 1000

        return EmbeddingResult(
            vector=self.normalize(vec),
            latency_ms=latency_ms,
            model_name=self.name,
            input_type="video",
        )

    def embed_text(self, text: str) -> EmbeddingResult:
        start = time.perf_counter()

        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)

        vec = outputs.squeeze().cpu().numpy().astype(np.float32)
        latency_ms = (time.perf_counter() - start) * 1000

        return EmbeddingResult(
            vector=self.normalize(vec),
            latency_ms=latency_ms,
            model_name=self.name,
            input_type="text",
        )
