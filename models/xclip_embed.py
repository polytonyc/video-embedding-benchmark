"""Microsoft X-CLIP adapter (local, open-source).

X-CLIP extends CLIP with temporal modeling via cross-frame attention,
making it a proper video-text contrastive model (not just frame averaging).

Note: X-CLIP's architecture requires both text and video in the forward pass
(text-conditioned video features). We use a neutral prompt for standalone
video embedding and the full model for text embedding.

Requires: pip install transformers torch opencv-python
GPU/MPS recommended, works on CPU.
"""

import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoTokenizer,
    VideoMAEImageProcessor,
    XCLIPModel,
)

from .base import BaseEmbedder, EmbeddingResult


def extract_pil_frames(video_path: Path, num_frames: int = 8) -> list[Image.Image]:
    """Extract evenly-spaced frames as PIL Images."""
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
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()

    while len(frames) < num_frames:
        frames.append(frames[-1])
    return frames[:num_frames]


class XCLIPEmbedder(BaseEmbedder):
    name = "xclip_base"
    display_name = "X-CLIP Base (temporal)"
    dimensions = 512
    is_api = False

    def __init__(self, num_frames: int = 8):
        self.num_frames = num_frames
        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        model_name = "microsoft/xclip-base-patch32"
        print(f"Loading X-CLIP Base on {self.device}...")
        self.model = XCLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.video_processor = VideoMAEImageProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Pre-compute neutral prompt for standalone video embedding
        self._neutral_text = self.tokenizer(
            ["a video"],
            return_tensors="pt",
            padding=True,
        )
        self._neutral_text = {k: v.to(self.device) for k, v in self._neutral_text.items()}

    def embed_video(self, video_path: Path) -> EmbeddingResult:
        start = time.perf_counter()

        frames = extract_pil_frames(video_path, self.num_frames)
        pixel_values = self.video_processor(frames, return_tensors="pt")["pixel_values"].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=self._neutral_text["input_ids"],
                attention_mask=self._neutral_text["attention_mask"],
                pixel_values=pixel_values,
            )
            vec = outputs.video_embeds.squeeze().cpu().numpy().astype(np.float32)

        latency_ms = (time.perf_counter() - start) * 1000

        return EmbeddingResult(
            vector=self.normalize(vec),
            latency_ms=latency_ms,
            model_name=self.name,
            input_type="video",
        )

    def embed_text(self, text: str) -> EmbeddingResult:
        start = time.perf_counter()

        text_inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        # X-CLIP needs video input too — use a dummy zero video
        dummy_video = torch.zeros(1, self.num_frames, 3, 224, 224, device=self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
                pixel_values=dummy_video,
            )
            # text_embeds shape is (batch, num_text, dim) — squeeze both
            vec = outputs.text_embeds.squeeze().cpu().numpy().astype(np.float32)

        latency_ms = (time.perf_counter() - start) * 1000

        return EmbeddingResult(
            vector=self.normalize(vec),
            latency_ms=latency_ms,
            model_name=self.name,
            input_type="text",
        )
