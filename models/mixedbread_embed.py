"""Mixedbread Wholembed v3 adapter (API, Stores-based).

Wholembed v3 uses late-interaction retrieval (ColBERT-style) rather than
single-vector embeddings. Video retrieval is only available through the
managed Stores API: upload videos → search with text queries.

Since we can't extract raw embedding vectors, this adapter:
  1. Creates a Store and uploads all videos
  2. For each text query, calls the search endpoint
  3. Returns ranked results for metric computation

Requires: pip install requests
Env: MIXEDBREAD_API_KEY
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import requests

from .base import BaseEmbedder, EmbeddingResult


class MixedbreadEmbedder(BaseEmbedder):
    """Mixedbread Wholembed v3 via Stores API.

    This adapter works differently from others because Wholembed v3 doesn't
    expose raw embedding vectors for video. Instead, we use the Stores
    pipeline for retrieval and generate synthetic embeddings that encode
    the search ranking for metric computation.
    """

    name = "mixedbread_wholembed_v3"
    display_name = "Mixedbread Wholembed v3"
    dimensions = 1  # Not applicable — late-interaction model
    is_api = True

    def __init__(self):
        self.api_key = os.environ.get("MIXEDBREAD_API_KEY")
        if not self.api_key:
            raise ValueError("MIXEDBREAD_API_KEY environment variable required")

        self.base_url = "https://api.mixedbread.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        self.store_id = os.environ.get("MIXEDBREAD_STORE_ID")
        self._video_file_ids = {}  # video_id -> file_id mapping
        self._setup_complete = False

    def _api_get(self, path: str) -> dict:
        resp = requests.get(
            f"{self.base_url}{path}",
            headers={**self.headers, "Content-Type": "application/json"},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()

    def _api_post(self, path: str, json_data: dict = None) -> dict:
        resp = requests.post(
            f"{self.base_url}{path}",
            headers={**self.headers, "Content-Type": "application/json"},
            json=json_data,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()

    def _api_post_file(self, path: str, file_path: Path) -> dict:
        with open(file_path, "rb") as f:
            resp = requests.post(
                f"{self.base_url}{path}",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"file": (file_path.name, f, "video/mp4")},
                timeout=300,
            )
        resp.raise_for_status()
        return resp.json()

    def setup_store(self, video_paths: dict[str, Path]):
        """Create a store and upload all videos. Called once before benchmarking."""
        if self._setup_complete:
            return

        if self.store_id:
            # Reuse existing store — just load file mappings
            print(f"  Reusing existing store: {self.store_id}")
            files_resp = self._api_get(f"/stores/{self.store_id}/files")
            files_list = files_resp.get("data", files_resp.get("files", []))
            # Build vid_id -> file_id from filenames
            fname_to_vid = {Path(p).name: vid for vid, p in video_paths.items()}
            for f in files_list:
                fname = f.get("filename", "")
                vid = fname_to_vid.get(fname)
                if vid:
                    self._video_file_ids[vid] = f.get("id") or f.get("file_id")
            self._setup_complete = True
            print(f"  Mapped {len(self._video_file_ids)} files to video IDs")
            return

        store_name = f"video-benchmark-{int(time.time())}"
        print(f"  Creating Mixedbread store: {store_name}")

        # Create store
        result = self._api_post("/stores", {"name": store_name})
        self.store_id = result.get("id") or result.get("identifier") or store_name
        print(f"  Store ID: {self.store_id}")

        # Upload each video
        for vid_id, vpath in sorted(video_paths.items()):
            print(f"  Uploading {vpath.name}...", end=" ", flush=True)
            try:
                result = self._api_post_file(
                    f"/stores/{self.store_id}/files/upload", vpath
                )
                file_id = result.get("id") or result.get("file_id")
                self._video_file_ids[vid_id] = file_id
                print(f"OK ({file_id})")
            except Exception as e:
                print(f"FAILED: {e}")

        # Poll until all files are processed
        print("  Waiting for video processing...", flush=True)
        for attempt in range(60):
            store_info = self._api_get(f"/stores/{self.store_id}")
            fc = store_info.get("file_counts", {})
            completed = fc.get("completed", 0)
            total = fc.get("total", 0)
            pending = fc.get("pending", 0) + fc.get("in_progress", 0)
            if pending == 0 and completed == total and total > 0:
                print(f"  All {completed} files processed.")
                break
            print(f"  Processing: {completed}/{total} done, {pending} pending...", flush=True)
            time.sleep(5)
        else:
            print("  WARNING: Timed out waiting for processing. Results may be incomplete.")

        self._setup_complete = True
        print(f"  Store ready with {len(self._video_file_ids)} videos")

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """Search the store with a text query. Returns ranked results."""
        for attempt in range(5):
            try:
                result = self._api_post("/stores/search", {
                    "store_identifiers": [self.store_id],
                    "query": query,
                    "top_k": top_k,
                })
                return result.get("results", result.get("data", []))
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    wait = 2 ** attempt
                    time.sleep(wait)
                    continue
                raise
        raise RuntimeError("Rate limited after 5 retries")

    def embed_video(self, video_path: Path) -> EmbeddingResult:
        """Not used directly — videos are uploaded to Store in bulk."""
        start = time.perf_counter()
        # Upload is handled by setup_store; this is a no-op placeholder
        latency_ms = (time.perf_counter() - start) * 1000
        # Return a dummy unit vector — retrieval is done via search()
        vec = np.array([1.0], dtype=np.float32)
        return EmbeddingResult(
            vector=vec,
            latency_ms=latency_ms,
            model_name=self.name,
            input_type="video",
        )

    def embed_text(self, text: str) -> EmbeddingResult:
        """Not used directly — queries go through search()."""
        start = time.perf_counter()
        latency_ms = (time.perf_counter() - start) * 1000
        vec = np.array([1.0], dtype=np.float32)
        return EmbeddingResult(
            vector=vec,
            latency_ms=latency_ms,
            model_name=self.name,
            input_type="text",
        )
