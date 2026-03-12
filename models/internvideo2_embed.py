"""InternVideo2-Stage2 adapter (local, open-source).

InternVideo2 is a video-text contrastive model from OpenGVLab with
true temporal understanding via a ViT backbone + BERT text encoder.
Produces 768-dim L2-normalized embeddings in a shared text-video space.

Uses the 6B variant which supports AutoModel.from_pretrained() with
trust_remote_code=True. Requires ~12GB RAM/VRAM.

Requires: pip install torch transformers einops timm opencv-python
GPU/MPS recommended, works on CPU (slowly).
"""

import time
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoModel

from .base import BaseEmbedder, EmbeddingResult

# ImageNet normalization constants
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def extract_frames_tensor(
    video_path: Path, num_frames: int = 4, size: int = 224
) -> torch.Tensor:
    """Extract uniformly-spaced frames and return as normalized tensor.

    Returns shape (1, num_frames, 3, size, size) — batch of 1 video.
    """
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise ValueError(f"Cannot read video: {video_path}")

    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # BGR -> RGB, resize
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (size, size))
            # Normalize
            frame = frame.astype(np.float32) / 255.0
            frame = (frame - _MEAN) / _STD
            # HWC -> CHW
            frame = frame.transpose(2, 0, 1)
            frames.append(frame)
    cap.release()

    # Pad if needed
    while len(frames) < num_frames:
        frames.append(frames[-1])
    frames = frames[:num_frames]

    # Stack to (1, T, C, H, W)
    tensor = torch.from_numpy(np.stack(frames)).unsqueeze(0)
    return tensor


class InternVideo2Embedder(BaseEmbedder):
    name = "internvideo2_6b"
    display_name = "InternVideo2-Stage2 6B"
    dimensions = 768
    is_api = False

    def __init__(self, num_frames: int = 4):
        self.num_frames = num_frames
        # InternVideo2 uses Conv3D (not supported on MPS) and hardcodes
        # some CUDA calls internally, so we must use CPU on non-CUDA systems
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_name = "OpenGVLab/InternVideo2-Stage2_6B"
        print(f"Loading InternVideo2-Stage2 6B on {self.device}...")
        print("  (This may take a while — 6B parameter model)")

        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.device = self.device  # Override hardcoded "cuda"
        self.model = AutoModel.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=True,
        ).to(self.device).eval()
        # Also patch the internal config reference
        self.model._config.device = self.device

    def embed_video(self, video_path: Path) -> EmbeddingResult:
        start = time.perf_counter()

        frames = extract_frames_tensor(video_path, self.num_frames).to(
            device=self.device, dtype=self.model.dtype if hasattr(self.model, 'dtype') else torch.float32
        )

        with torch.no_grad():
            vec = self.model.get_vid_feat(frames)
            vec = vec.squeeze().cpu().numpy().astype(np.float32)

        latency_ms = (time.perf_counter() - start) * 1000

        return EmbeddingResult(
            vector=self.normalize(vec),
            latency_ms=latency_ms,
            model_name=self.name,
            input_type="video",
        )

    def embed_text(self, text: str) -> EmbeddingResult:
        start = time.perf_counter()

        with torch.no_grad():
            vec = self.model.get_txt_feat(text)
            vec = vec.squeeze().cpu().numpy().astype(np.float32)

        latency_ms = (time.perf_counter() - start) * 1000

        return EmbeddingResult(
            vector=self.normalize(vec),
            latency_ms=latency_ms,
            model_name=self.name,
            input_type="text",
        )
