"""
Download a curated video dataset for embedding benchmarking.

Uses direct-download CC0/public-domain videos from Pexels (permanent URLs).
All videos are normalized to consistent characteristics:
  - Duration: 5-15 seconds
  - Resolution: 640x360 (360p) minimum
  - Format: MP4 / H.264
  - Audio: ambient/natural (not music-overlaid)

Dataset design follows IR evaluation best practices:
  - 5 semantic categories, 4 videos each (20 videos)
  - 3 query types per video (exact, partial, hard-negative)
  - Graded relevance: 2 (exact), 1 (partial), 0 (irrelevant)
  - Hard negatives are semantically adjacent (same broad domain)

Usage:
    python download_dataset.py
"""

import json
import subprocess
import sys
from pathlib import Path

import requests
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"
VIDEO_DIR = DATA_DIR / "videos"
QUERIES_FILE = DATA_DIR / "queries.json"

# Direct-download CC0 videos from Pexels (permanent CDN URLs).
# Each entry: (filename, download_url, duration_seconds, description)
# All selected to be 5-15s, landscape, natural ambient audio, no text overlays.
VIDEO_SOURCES = {
    "sports": [
        {
            "filename": "sports_basketball.mp4",
            "url": "https://videos.pexels.com/video-files/3191572/3191572-sd_640_360_25fps.mp4",
            "description": "Basketball players practicing layups and dribbling on an outdoor court",
        },
        {
            "filename": "sports_soccer.mp4",
            "url": "https://videos.pexels.com/video-files/2657257/2657257-sd_640_360_24fps.mp4",
            "description": "Soccer players kicking a ball on a green grass field",
        },
        {
            "filename": "sports_swimming.mp4",
            "url": "https://videos.pexels.com/video-files/5264548/5264548-sd_640_360_25fps.mp4",
            "description": "A swimmer doing freestyle strokes in a swimming pool with lane dividers",
        },
        {
            "filename": "sports_running.mp4",
            "url": "https://videos.pexels.com/video-files/3209828/3209828-sd_640_360_25fps.mp4",
            "description": "Runners sprinting on an athletic track during a race",
        },
    ],
    "cooking": [
        {
            "filename": "cooking_chopping.mp4",
            "url": "https://videos.pexels.com/video-files/3195394/3195394-sd_640_360_25fps.mp4",
            "description": "A person chopping vegetables with a knife on a wooden cutting board",
        },
        {
            "filename": "cooking_stirring.mp4",
            "url": "https://videos.pexels.com/video-files/854565/854565-sd_640_360_25fps.mp4",
            "description": "Stirring food in a pot on a stovetop in a home kitchen",
        },
        {
            "filename": "cooking_pouring.mp4",
            "url": "https://videos.pexels.com/video-files/3196175/3196175-sd_640_360_25fps.mp4",
            "description": "Pouring liquid batter into a baking pan for a cake",
        },
        {
            "filename": "cooking_plating.mp4",
            "url": "https://videos.pexels.com/video-files/3752411/3752411-sd_640_360_24fps.mp4",
            "description": "A chef carefully arranging food on a plate in a restaurant kitchen",
        },
    ],
    "nature": [
        {
            "filename": "nature_waterfall.mp4",
            "url": "https://videos.pexels.com/video-files/855668/855668-sd_640_360_30fps.mp4",
            "description": "A waterfall cascading over rocks surrounded by green vegetation",
        },
        {
            "filename": "nature_ocean.mp4",
            "url": "https://videos.pexels.com/video-files/1093662/1093662-sd_640_360_30fps.mp4",
            "description": "Ocean waves rolling and crashing onto a sandy beach shore",
        },
        {
            "filename": "nature_forest.mp4",
            "url": "https://videos.pexels.com/video-files/3571264/3571264-sd_640_360_30fps.mp4",
            "description": "Sunlight filtering through tall trees in a dense green forest",
        },
        {
            "filename": "nature_birds.mp4",
            "url": "https://videos.pexels.com/video-files/4457089/4457089-sd_640_360_30fps.mp4",
            "description": "Birds perched on branches and flying between trees",
        },
    ],
    "urban": [
        {
            "filename": "urban_traffic.mp4",
            "url": "https://videos.pexels.com/video-files/2053100/2053100-sd_640_360_30fps.mp4",
            "description": "Cars and buses moving through a busy city street intersection",
        },
        {
            "filename": "urban_pedestrians.mp4",
            "url": "https://videos.pexels.com/video-files/3121459/3121459-sd_640_360_24fps.mp4",
            "description": "Pedestrians walking on a crowded city sidewalk near shops",
        },
        {
            "filename": "urban_construction.mp4",
            "url": "https://videos.pexels.com/video-files/855271/855271-sd_640_360_25fps.mp4",
            "description": "A construction site with cranes and scaffolding on a building",
        },
        {
            "filename": "urban_skyline.mp4",
            "url": "https://videos.pexels.com/video-files/2795391/2795391-sd_640_360_25fps.mp4",
            "description": "A city skyline with tall buildings seen from a distance at dusk",
        },
    ],
    "technology": [
        {
            "filename": "tech_typing.mp4",
            "url": "https://videos.pexels.com/video-files/5496001/5496001-sd_640_360_24fps.mp4",
            "description": "Hands typing rapidly on a laptop keyboard at a desk",
        },
        {
            "filename": "tech_soldering.mp4",
            "url": "https://videos.pexels.com/video-files/7230998/7230998-sd_640_360_25fps.mp4",
            "description": "A person soldering electronic components onto a circuit board",
        },
        {
            "filename": "tech_3dprinter.mp4",
            "url": "https://videos.pexels.com/video-files/855255/855255-sd_640_360_25fps.mp4",
            "description": "A 3D printer extruding filament to create a plastic object",
        },
        {
            "filename": "tech_robotics.mp4",
            "url": "https://videos.pexels.com/video-files/8084614/8084614-sd_640_360_25fps.mp4",
            "description": "A small robot moving and responding on a tabletop",
        },
    ],
}

# Query design follows graded relevance (NDCG-style):
#   relevance=2: Exact semantic match (query describes this specific video)
#   relevance=1: Partial match (query describes the category but not this exact scene)
#   relevance=0: Hard negative (semantically adjacent domain, but different content)
#
# Hard negatives are intentionally tricky — same broad theme (e.g., "motion" or
# "hands doing something") to test whether the model captures fine-grained semantics.


def build_queries() -> dict:
    """
    Build ground-truth queries with graded relevance labels.

    Returns {videos: {...}, queries: [...], relevance_matrix: {...}}
    """
    videos = {}
    queries = []
    # Full relevance matrix: query_id -> {video_id: relevance_grade}
    relevance_matrix = {}

    vid_idx = 0
    categories = list(VIDEO_SOURCES.keys())

    for cat_idx, (category, items) in enumerate(VIDEO_SOURCES.items()):
        for item_idx, item in enumerate(items):
            vid = f"v{vid_idx:02d}"
            videos[vid] = {
                "filename": item["filename"],
                "category": category,
                "description": item["description"],
            }

            same_cat_vids = [f"v{vid_idx - item_idx + j:02d}" for j in range(4)]

            # --- Query 1: Exact match ---
            qid_exact = f"q{len(queries):03d}"
            # Build full relevance for this query
            rel = {}
            rel[vid] = 2  # exact match
            for sv in same_cat_vids:
                if sv != vid:
                    rel[sv] = 1  # same category = partial
            # All other videos = 0 (implicit)

            queries.append({
                "query_id": qid_exact,
                "text": item["description"],
                "type": "exact",
                "target_video": vid,
                "category": category,
            })
            relevance_matrix[qid_exact] = rel

            # --- Query 2: Category-level (partial) ---
            category_queries = {
                "sports": "People playing a competitive sport outdoors",
                "cooking": "Someone preparing food in a kitchen",
                "nature": "A natural landscape with water or vegetation",
                "urban": "A busy scene in a modern city",
                "technology": "Someone working with electronic equipment or computers",
            }
            qid_partial = f"q{len(queries):03d}"
            rel_partial = {sv: 1 for sv in same_cat_vids}
            rel_partial[vid] = 2  # slightly more relevant since we're testing from this vid's perspective

            queries.append({
                "query_id": qid_partial,
                "text": category_queries[category],
                "type": "partial",
                "target_video": vid,
                "category": category,
            })
            relevance_matrix[qid_partial] = rel_partial

            # --- Query 3: Hard negative ---
            # Pick a semantically adjacent category for the negative
            adjacent = {
                "sports": "cooking",    # both involve people doing physical actions
                "cooking": "technology", # both involve hands working with tools
                "nature": "urban",       # both are environments/scenes
                "urban": "nature",       # both are environments/scenes
                "technology": "cooking", # both involve hands working with tools/objects
            }
            neg_category = adjacent[category]
            neg_queries = {
                ("sports", "cooking"): "A chef rapidly slicing ingredients with a sharp knife",
                ("cooking", "technology"): "A technician carefully assembling small electronic parts by hand",
                ("nature", "urban"): "A quiet residential street with houses and parked cars",
                ("urban", "nature"): "A calm lake reflecting mountains in the distance",
                ("technology", "cooking"): "A baker using a mixer to blend ingredients in a bowl",
            }
            qid_neg = f"q{len(queries):03d}"
            neg_cat_start = categories.index(neg_category) * 4
            neg_cat_vids = [f"v{neg_cat_start + j:02d}" for j in range(4)]
            rel_neg = {nv: 1 for nv in neg_cat_vids}
            # The target video from current category should NOT be relevant

            queries.append({
                "query_id": qid_neg,
                "text": neg_queries[(category, neg_category)],
                "type": "hard_negative",
                "target_video": vid,  # this video should NOT rank high
                "expected_category": neg_category,
                "category": category,
            })
            relevance_matrix[qid_neg] = rel_neg

            vid_idx += 1

    return {
        "videos": videos,
        "queries": queries,
        "relevance_matrix": relevance_matrix,
        "metadata": {
            "num_videos": len(videos),
            "num_queries": len(queries),
            "categories": categories,
            "relevance_grades": {
                "2": "exact match",
                "1": "partial / same-category match",
                "0": "irrelevant (implicit — not in relevance dict)",
            },
            "video_characteristics": {
                "resolution": "640x360 (360p)",
                "duration": "5-15 seconds",
                "format": "MP4 / H.264",
                "audio": "ambient / natural",
            },
        },
    }


def download_video(url: str, dest: Path, max_retries: int = 2) -> bool:
    """Download a video, retrying on failure."""
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, stream=True, timeout=30)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True,
                desc=dest.name, leave=False,
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
            # Verify it's a valid video (at least 10KB)
            if dest.stat().st_size < 10_000:
                print(f"  Warning: {dest.name} is suspiciously small ({dest.stat().st_size} bytes)")
                dest.unlink()
                continue
            return True
        except Exception as e:
            if attempt < max_retries:
                print(f"  Retry {attempt + 1} for {dest.name}: {e}")
            else:
                print(f"  Failed to download {dest.name}: {e}")
    return False


def normalize_videos():
    """
    Normalize all downloaded videos to uniform characteristics using ffmpeg.
    Ensures: 360p, 10s max, H.264, 24fps, mono audio.
    """
    if not any(VIDEO_DIR.glob("*.mp4")):
        return

    has_ffmpeg = subprocess.run(
        ["ffmpeg", "-version"], capture_output=True
    ).returncode == 0

    if not has_ffmpeg:
        print("\nffmpeg not found — skipping normalization. Videos may vary in specs.")
        return

    print("\nNormalizing videos to uniform characteristics...")
    for mp4 in sorted(VIDEO_DIR.glob("*.mp4")):
        tmp = mp4.with_suffix(".tmp.mp4")
        cmd = [
            "ffmpeg", "-y", "-i", str(mp4),
            "-t", "10",              # max 10 seconds
            "-vf", "scale=640:360",  # 360p
            "-r", "24",              # 24 fps
            "-c:v", "libx264",       # H.264
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",           # AAC audio
            "-ac", "1",              # mono
            "-b:a", "64k",
            "-movflags", "+faststart",
            str(tmp),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            tmp.replace(mp4)
            print(f"  Normalized: {mp4.name}")
        else:
            tmp.unlink(missing_ok=True)
            print(f"  Failed to normalize {mp4.name}: {result.stderr[:200]}")


def main():
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    # Build and write queries
    dataset = build_queries()
    with open(QUERIES_FILE, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Wrote {QUERIES_FILE}")
    print(f"  {dataset['metadata']['num_videos']} videos, {dataset['metadata']['num_queries']} queries")

    # Download videos
    print(f"\nDownloading videos to {VIDEO_DIR}/\n")
    downloaded = 0
    failed = []

    for category, items in VIDEO_SOURCES.items():
        print(f"[{category}]")
        for item in items:
            dest = VIDEO_DIR / item["filename"]
            if dest.exists() and dest.stat().st_size > 10_000:
                print(f"  {item['filename']} — exists, skipping")
                downloaded += 1
                continue

            if download_video(item["url"], dest):
                downloaded += 1
            else:
                failed.append(item["filename"])

    print(f"\nDownloaded: {downloaded}/20")
    if failed:
        print(f"Failed: {', '.join(failed)}")

    # Normalize to uniform specs
    normalize_videos()

    # Print video file sizes
    print("\nVideo files:")
    total_bytes = 0
    for mp4 in sorted(VIDEO_DIR.glob("*.mp4")):
        size = mp4.stat().st_size
        total_bytes += size
        print(f"  {mp4.name}: {size / 1024:.0f} KB")
    print(f"  Total: {total_bytes / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
