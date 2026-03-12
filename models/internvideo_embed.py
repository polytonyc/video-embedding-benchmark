"""InternVideo2.5 adapter (local, open-source).

Requires: pip install transformers torch decord
GPU recommended. Falls back to CPU if no CUDA available.
"""

import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .base import BaseEmbedder, EmbeddingResult

# Video frame extraction
try:
    from decord import VideoReader, cpu as decord_cpu
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False
    import cv2


def extract_frames(video_path: Path, num_frames: int = 8) -> np.ndarray:
    """Extract evenly-spaced frames from a video. Returns (N, H, W, 3) uint8."""
    if HAS_DECORD:
        vr = VideoReader(str(video_path), ctx=decord_cpu(0))
        total = len(vr)
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()
    else:
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        frames = np.stack(frames)
    return frames


class InternVideoEmbedder(BaseEmbedder):
    name = "internvideo2_5"
    display_name = "InternVideo2.5"
    dimensions = 768
    is_api = False

    def __init__(self, num_frames: int = 8):
        self.num_frames = num_frames
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        # Use InternVideo2-Stage2_1B-224p-f4 (smaller, more practical variant)
        model_name = "OpenGVLab/InternVideo2-Stage2_1B-224p-f4"
        print(f"Loading InternVideo2 from {model_name} on {self.device}...")

        try:
            self.model = AutoModel.from_pretrained(
                model_name, trust_remote_code=True
            ).to(self.device).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            self._loaded = True
        except Exception as e:
            print(f"Warning: Could not load InternVideo2 model: {e}")
            print("Install with: pip install transformers torch decord")
            print("Model will be downloaded on first use (~2GB)")
            self._loaded = False

    def _preprocess_video(self, video_path: Path) -> torch.Tensor:
        """Extract frames and preprocess for the model."""
        frames = extract_frames(video_path, self.num_frames)
        # Normalize to [0, 1] and resize to 224x224
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        processed = torch.stack([transform(f) for f in frames])
        # Shape: (num_frames, 3, 224, 224) → (1, num_frames, 3, 224, 224)
        return processed.unsqueeze(0).to(self.device)

    def embed_video(self, video_path: Path) -> EmbeddingResult:
        if not self._loaded:
            raise RuntimeError("InternVideo2 model not loaded")

        start = time.perf_counter()
        video_tensor = self._preprocess_video(video_path)

        with torch.no_grad():
            output = self.model.encode_video(video_tensor)
            if isinstance(output, tuple):
                output = output[0]
            vec = output.squeeze().cpu().numpy().astype(np.float32)

        latency_ms = (time.perf_counter() - start) * 1000

        return EmbeddingResult(
            vector=self.normalize(vec),
            latency_ms=latency_ms,
            model_name=self.name,
            input_type="video",
        )

    def embed_text(self, text: str) -> EmbeddingResult:
        if not self._loaded:
            raise RuntimeError("InternVideo2 model not loaded")

        start = time.perf_counter()
        tokens = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=77
        ).to(self.device)

        with torch.no_grad():
            output = self.model.encode_text(tokens)
            if isinstance(output, tuple):
                output = output[0]
            vec = output.squeeze().cpu().numpy().astype(np.float32)

        latency_ms = (time.perf_counter() - start) * 1000

        return EmbeddingResult(
            vector=self.normalize(vec),
            latency_ms=latency_ms,
            model_name=self.name,
            input_type="text",
        )
