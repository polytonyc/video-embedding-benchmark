"""Amazon Nova Multimodal Embeddings adapter.

Supports two auth methods:
  1. Bearer token: export AWS_BEARER_TOKEN_BEDROCK="bedrock-api-key-..."
  2. Standard IAM: export AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY

Requires: pip install requests (bearer token) or boto3 (IAM)
"""

import base64
import json
import os
import time
from pathlib import Path

import numpy as np
import requests

from .base import BaseEmbedder, EmbeddingResult


class NovaEmbedder(BaseEmbedder):
    name = "nova_multimodal"
    display_name = "Amazon Nova Multimodal"
    dimensions = 1024
    is_api = True

    def __init__(self, dimensions: int = 1024):
        self.dimensions = dimensions
        self.model_id = "amazon.nova-2-multimodal-embeddings-v1:0"
        self.region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

        # Prefer bearer token auth (Bedrock API key)
        self.bearer_token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        if self.bearer_token:
            self.endpoint = (
                f"https://bedrock-runtime.{self.region}.amazonaws.com"
                f"/model/{self.model_id}/invoke"
            )
            self._invoke = self._invoke_bearer
        else:
            try:
                import boto3
            except ImportError:
                raise ImportError(
                    "Set AWS_BEARER_TOKEN_BEDROCK or install boto3 with IAM credentials"
                )
            self.client = boto3.client("bedrock-runtime", region_name=self.region)
            self._invoke = self._invoke_boto3

    def _invoke_bearer(self, body: str) -> dict:
        resp = requests.post(
            self.endpoint,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self.bearer_token}",
            },
            data=body,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()

    def _invoke_boto3(self, body: str) -> dict:
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        return json.loads(response["body"].read())

    def embed_video(self, video_path: Path) -> EmbeddingResult:
        with open(video_path, "rb") as f:
            video_bytes = f.read()

        video_b64 = base64.b64encode(video_bytes).decode("utf-8")
        suffix = video_path.suffix.lower()
        media_types = {".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm"}
        media_type = media_types.get(suffix, "video/mp4")

        body = json.dumps({
            "inputVideo": {
                "source": {"bytes": video_b64},
                "mediaType": media_type,
            },
            "embeddingConfig": {
                "outputEmbeddingLength": self.dimensions,
            },
        })

        start = time.perf_counter()
        result = self._invoke(body)
        latency_ms = (time.perf_counter() - start) * 1000

        vec = np.array(result["embedding"], dtype=np.float32)

        return EmbeddingResult(
            vector=self.normalize(vec),
            latency_ms=latency_ms,
            model_name=self.name,
            input_type="video",
        )

    def embed_text(self, text: str) -> EmbeddingResult:
        body = json.dumps({
            "inputText": text,
            "embeddingConfig": {
                "outputEmbeddingLength": self.dimensions,
            },
        })

        start = time.perf_counter()
        result = self._invoke(body)
        latency_ms = (time.perf_counter() - start) * 1000

        vec = np.array(result["embedding"], dtype=np.float32)

        return EmbeddingResult(
            vector=self.normalize(vec),
            latency_ms=latency_ms,
            model_name=self.name,
            input_type="text",
        )
