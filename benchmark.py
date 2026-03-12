"""
Video Embedding Model Benchmark
================================

Evaluates text-to-video retrieval using standard IR metrics with graded relevance.

Methodology:
  1. Embed all 20 videos with each model
  2. Embed all 60 text queries with each model
  3. Rank videos by cosine similarity to each query
  4. Compute metrics against ground-truth relevance judgments

Metrics (following BEIR / MTEB conventions):
  - NDCG@K (K=1,5,10): Normalized Discounted Cumulative Gain with graded relevance
  - Recall@K (K=1,5,10): Fraction of relevant docs retrieved in top K
  - MRR: Mean Reciprocal Rank of first relevant result
  - MAP@10: Mean Average Precision at 10
  - Precision@5: Fraction of top-5 that are relevant

Breakdown by query difficulty:
  - exact: Query describes a specific video
  - partial: Query describes a category
  - hard_negative: Query from adjacent domain (should NOT match target)

Usage:
    python benchmark.py --all
    python benchmark.py --model gemini
    python benchmark.py --model siglip
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
VIDEO_DIR = DATA_DIR / "videos"
QUERIES_FILE = DATA_DIR / "queries.json"

SEED = 42
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# IR Metrics (graded relevance)
# ---------------------------------------------------------------------------

def dcg_at_k(relevances: list[int], k: int) -> float:
    """Discounted Cumulative Gain at K."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += (2 ** rel - 1) / math.log2(i + 2)  # i+2 because log2(1)=0
    return dcg


def ndcg_at_k(relevances: list[int], k: int) -> float:
    """Normalized DCG at K."""
    dcg = dcg_at_k(relevances, k)
    ideal = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def recall_at_k(relevances: list[int], k: int, total_relevant: int) -> float:
    """Recall at K (binary: any relevance > 0 counts)."""
    if total_relevant == 0:
        return 0.0
    retrieved_relevant = sum(1 for r in relevances[:k] if r > 0)
    return retrieved_relevant / total_relevant


def precision_at_k(relevances: list[int], k: int) -> float:
    """Precision at K."""
    return sum(1 for r in relevances[:k] if r > 0) / k


def average_precision(relevances: list[int], k: int) -> float:
    """Average Precision at K."""
    total_relevant = sum(1 for r in relevances if r > 0)
    if total_relevant == 0:
        return 0.0
    hits = 0
    sum_prec = 0.0
    for i, rel in enumerate(relevances[:k]):
        if rel > 0:
            hits += 1
            sum_prec += hits / (i + 1)
    return sum_prec / min(total_relevant, k)


def reciprocal_rank(relevances: list[int]) -> float:
    """Reciprocal rank of first relevant result."""
    for i, rel in enumerate(relevances):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


def compute_all_metrics(
    query_vectors: dict[str, np.ndarray],
    video_vectors: dict[str, np.ndarray],
    queries: list[dict],
    relevance_matrix: dict[str, dict[str, int]],
) -> dict:
    """Compute all retrieval metrics."""
    video_ids = sorted(video_vectors.keys())
    video_matrix = np.stack([video_vectors[vid] for vid in video_ids])

    all_metrics = {k: [] for k in [
        "ndcg@1", "ndcg@5", "ndcg@10",
        "recall@1", "recall@5", "recall@10",
        "precision@5", "map@10", "mrr",
    ]}
    by_type = {}
    per_query = []

    for q in queries:
        qid = q["query_id"]
        qtype = q["type"]
        if qid not in query_vectors:
            continue

        q_vec = query_vectors[qid]
        rel_dict = relevance_matrix.get(qid, {})
        total_relevant = sum(1 for v in rel_dict.values() if v > 0)

        # Compute similarities and rank
        sims = video_matrix @ q_vec
        ranked_indices = np.argsort(-sims)
        ranked_ids = [video_ids[i] for i in ranked_indices]
        ranked_sims = [float(sims[i]) for i in ranked_indices]

        # Build relevance list in ranked order
        relevances = [rel_dict.get(vid, 0) for vid in ranked_ids]

        # Compute metrics
        m = {
            "ndcg@1": ndcg_at_k(relevances, 1),
            "ndcg@5": ndcg_at_k(relevances, 5),
            "ndcg@10": ndcg_at_k(relevances, 10),
            "recall@1": recall_at_k(relevances, 1, total_relevant),
            "recall@5": recall_at_k(relevances, 5, total_relevant),
            "recall@10": recall_at_k(relevances, 10, total_relevant),
            "precision@5": precision_at_k(relevances, 5),
            "map@10": average_precision(relevances, 10),
            "mrr": reciprocal_rank(relevances),
        }

        for k, v in m.items():
            all_metrics[k].append(v)

        if qtype not in by_type:
            by_type[qtype] = {k: [] for k in all_metrics}
        for k, v in m.items():
            by_type[qtype][k].append(v)

        per_query.append({
            "query_id": qid,
            "text": q["text"],
            "type": qtype,
            "target_video": q.get("target_video"),
            "metrics": m,
            "top_5_videos": ranked_ids[:5],
            "top_5_similarities": ranked_sims[:5],
            "top_5_relevances": relevances[:5],
        })

    # Aggregate
    aggregated = {}
    for k, vals in all_metrics.items():
        if vals:
            aggregated[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "n": len(vals),
            }

    type_aggregated = {}
    for qtype, type_metrics in by_type.items():
        type_aggregated[qtype] = {}
        for k, vals in type_metrics.items():
            if vals:
                type_aggregated[qtype][k] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "n": len(vals),
                }

    return {
        "aggregated": aggregated,
        "by_query_type": type_aggregated,
        "per_query": per_query,
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(name: str):
    if name == "gemini":
        from models.gemini_embed import GeminiEmbedder
        return GeminiEmbedder()
    elif name == "marengo":
        from models.marengo_embed import MarengoEmbedder
        return MarengoEmbedder()
    elif name == "xclip":
        from models.xclip_embed import XCLIPEmbedder
        return XCLIPEmbedder()
    elif name == "nova":
        from models.nova_embed import NovaEmbedder
        return NovaEmbedder()
    elif name == "siglip":
        from models.siglip_embed import SigLIPEmbedder
        return SigLIPEmbedder()
    elif name == "internvideo2":
        from models.internvideo2_embed import InternVideo2Embedder
        return InternVideo2Embedder()
    elif name == "mixedbread":
        from models.mixedbread_embed import MixedbreadEmbedder
        return MixedbreadEmbedder()
    else:
        raise ValueError(f"Unknown model: {name}. Choose from: gemini, marengo, xclip, nova, siglip, internvideo2, mixedbread")


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(model_name: str) -> dict:
    with open(QUERIES_FILE) as f:
        dataset = json.load(f)

    videos = dataset["videos"]
    queries = dataset["queries"]
    relevance_matrix = dataset["relevance_matrix"]

    # Check videos exist
    missing = [
        info["filename"] for vid, info in videos.items()
        if not (VIDEO_DIR / info["filename"]).exists()
    ]
    if missing:
        print(f"Missing {len(missing)} videos. Run download_dataset.py first.")
        print(f"  {', '.join(missing[:5])}")
        sys.exit(1)

    # Load model
    print(f"\n{'=' * 60}")
    print(f"  Benchmarking: {model_name}")
    print(f"{'=' * 60}")
    model = load_model(model_name)
    print(f"  Model: {model.display_name}")
    print(f"  Dimensions: {model.dimensions}")
    print(f"  Type: {'API' if model.is_api else 'Local'}")

    # Embed videos
    print(f"\nEmbedding {len(videos)} videos...")
    video_vectors = {}
    video_latencies = []
    errors = []

    for vid, info in tqdm(sorted(videos.items()), desc="Videos"):
        vpath = VIDEO_DIR / info["filename"]
        try:
            result = model.embed_video(vpath)
            video_vectors[vid] = result.vector
            video_latencies.append(result.latency_ms)
        except Exception as e:
            print(f"\n  Error on {info['filename']}: {e}")
            errors.append({"video": vid, "error": str(e)})

    # Embed queries
    print(f"\nEmbedding {len(queries)} queries...")
    query_vectors = {}
    query_latencies = []

    for q in tqdm(queries, desc="Queries"):
        try:
            result = model.embed_text(q["text"])
            query_vectors[q["query_id"]] = result.vector
            query_latencies.append(result.latency_ms)
        except Exception as e:
            print(f"\n  Error on query {q['query_id']}: {e}")
            errors.append({"query": q["query_id"], "error": str(e)})

    if not video_vectors or not query_vectors:
        print("No embeddings produced. Check errors above.")
        return {}

    # Compute metrics
    print("\nComputing retrieval metrics...")
    results = compute_all_metrics(query_vectors, video_vectors, queries, relevance_matrix)

    # Add latency stats
    results["latency"] = {
        "video_embed_ms": {
            "median": float(np.median(video_latencies)) if video_latencies else 0,
            "p95": float(np.percentile(video_latencies, 95)) if video_latencies else 0,
            "mean": float(np.mean(video_latencies)) if video_latencies else 0,
            "total": float(np.sum(video_latencies)) if video_latencies else 0,
        },
        "text_embed_ms": {
            "median": float(np.median(query_latencies)) if query_latencies else 0,
            "p95": float(np.percentile(query_latencies, 95)) if query_latencies else 0,
            "mean": float(np.mean(query_latencies)) if query_latencies else 0,
        },
    }

    results["model_info"] = {
        "name": model.name,
        "display_name": model.display_name,
        "dimensions": model.dimensions,
        "is_api": model.is_api,
        "vector_bytes": model.dimensions * 4,
        "videos_embedded": len(video_vectors),
        "queries_embedded": len(query_vectors),
    }

    results["errors"] = errors

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{model.name}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    agg = results["aggregated"]
    print(f"\n{'=' * 60}")
    print(f"  {model.display_name} — Results")
    print(f"{'=' * 60}")

    header = f"  {'Metric':<16} {'Overall':>10}"
    types = list(results["by_query_type"].keys())
    for t in types:
        header += f" {t:>14}"
    print(header)
    print(f"  {'-' * (16 + 10 + 14 * len(types) + len(types) + 1)}")

    for metric in ["ndcg@1", "ndcg@5", "ndcg@10", "recall@1", "recall@5", "mrr", "map@10"]:
        if metric not in agg:
            continue
        row = f"  {metric:<16} {agg[metric]['mean']:>10.3f}"
        for t in types:
            tm = results["by_query_type"].get(t, {}).get(metric, {})
            row += f" {tm.get('mean', 0):>14.3f}"
        print(row)

    vlat = results["latency"]["video_embed_ms"]
    print(f"\n  Video embed latency: {vlat['median']:.0f}ms median, {vlat['p95']:.0f}ms p95")
    print(f"  Vector size: {results['model_info']['vector_bytes']:,} bytes ({model.dimensions}d float32)")
    if errors:
        print(f"  Errors: {len(errors)}")
    print(f"\n  Saved: {out}")

    return results


def run_stores_benchmark(model_name: str) -> dict:
    """Benchmark for Stores-based models (e.g., Mixedbread) that use upload+search
    rather than embed+rank."""
    with open(QUERIES_FILE) as f:
        dataset = json.load(f)

    videos = dataset["videos"]
    queries = dataset["queries"]
    relevance_matrix = dataset["relevance_matrix"]

    missing = [
        info["filename"] for vid, info in videos.items()
        if not (VIDEO_DIR / info["filename"]).exists()
    ]
    if missing:
        print(f"Missing {len(missing)} videos. Run download_dataset.py first.")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"  Benchmarking: {model_name} (Stores-based)")
    print(f"{'=' * 60}")

    model = load_model(model_name)
    print(f"  Model: {model.display_name}")
    print(f"  Type: Stores API (late-interaction retrieval)")

    # Upload all videos to the store
    video_paths = {vid: VIDEO_DIR / info["filename"] for vid, info in videos.items()}
    upload_start = time.perf_counter()
    model.setup_store(video_paths)
    upload_latency = (time.perf_counter() - upload_start) * 1000

    # Map filenames back to video IDs for result matching
    filename_to_vid = {info["filename"]: vid for vid, info in videos.items()}
    fileid_to_vid = {fid: vid for vid, fid in model._video_file_ids.items()}
    video_ids = sorted(videos.keys())

    # Run each query through search
    print(f"\nSearching {len(queries)} queries...")
    all_metrics = {k: [] for k in [
        "ndcg@1", "ndcg@5", "ndcg@10",
        "recall@1", "recall@5", "recall@10",
        "precision@5", "map@10", "mrr",
    ]}
    by_type = {}
    per_query = []
    query_latencies = []
    errors = []

    for q in tqdm(queries, desc="Queries"):
        qid = q["query_id"]
        qtype = q["type"]
        rel_dict = relevance_matrix.get(qid, {})
        total_relevant = sum(1 for v in rel_dict.values() if v > 0)

        try:
            start = time.perf_counter()
            results_list = model.search(q["text"], top_k=20)
            query_latency = (time.perf_counter() - start) * 1000
            query_latencies.append(query_latency)
        except Exception as e:
            print(f"\n  Error on query {qid}: {e}")
            errors.append({"query": qid, "error": str(e)})
            continue

        # Map search results to video IDs
        ranked_ids = []
        for r in results_list:
            # Try to match result to a video ID by filename
            fname = r.get("file_name", r.get("filename", ""))
            vid = filename_to_vid.get(fname)
            if not vid:
                # Try matching by file_id
                file_id = r.get("file_id", r.get("id", ""))
                vid = fileid_to_vid.get(file_id)
            if vid and vid not in ranked_ids:
                ranked_ids.append(vid)

        # Fill remaining with unranked videos
        for vid in video_ids:
            if vid not in ranked_ids:
                ranked_ids.append(vid)

        relevances = [rel_dict.get(vid, 0) for vid in ranked_ids]

        m = {
            "ndcg@1": ndcg_at_k(relevances, 1),
            "ndcg@5": ndcg_at_k(relevances, 5),
            "ndcg@10": ndcg_at_k(relevances, 10),
            "recall@1": recall_at_k(relevances, 1, total_relevant),
            "recall@5": recall_at_k(relevances, 5, total_relevant),
            "recall@10": recall_at_k(relevances, 10, total_relevant),
            "precision@5": precision_at_k(relevances, 5),
            "map@10": average_precision(relevances, 10),
            "mrr": reciprocal_rank(relevances),
        }

        for k, v in m.items():
            all_metrics[k].append(v)

        if qtype not in by_type:
            by_type[qtype] = {k: [] for k in all_metrics}
        for k, v in m.items():
            by_type[qtype][k].append(v)

        per_query.append({
            "query_id": qid,
            "text": q["text"],
            "type": qtype,
            "target_video": q.get("target_video"),
            "metrics": m,
            "top_5_videos": ranked_ids[:5],
            "top_5_relevances": relevances[:5],
        })

    if not per_query:
        print("No queries succeeded. Check errors above.")
        return {}

    # Aggregate
    aggregated = {}
    for k, vals in all_metrics.items():
        if vals:
            aggregated[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "n": len(vals),
            }

    type_aggregated = {}
    for qtype, type_metrics in by_type.items():
        type_aggregated[qtype] = {}
        for k, vals in type_metrics.items():
            if vals:
                type_aggregated[qtype][k] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "n": len(vals),
                }

    results = {
        "aggregated": aggregated,
        "by_query_type": type_aggregated,
        "per_query": per_query,
        "latency": {
            "video_embed_ms": {
                "median": upload_latency / len(videos),
                "p95": upload_latency / len(videos),
                "mean": upload_latency / len(videos),
                "total": upload_latency,
            },
            "text_embed_ms": {
                "median": float(np.median(query_latencies)) if query_latencies else 0,
                "p95": float(np.percentile(query_latencies, 95)) if query_latencies else 0,
                "mean": float(np.mean(query_latencies)) if query_latencies else 0,
            },
        },
        "model_info": {
            "name": model.name,
            "display_name": model.display_name,
            "dimensions": model.dimensions,
            "is_api": model.is_api,
            "vector_bytes": 0,  # late-interaction, no single vector
            "videos_embedded": len(model._video_file_ids),
            "queries_embedded": len(per_query),
            "note": "Stores-based retrieval (ColBERT-style late interaction), not single-vector embedding",
        },
        "errors": errors,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{model.name}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)

    agg = results["aggregated"]
    print(f"\n{'=' * 60}")
    print(f"  {model.display_name} — Results")
    print(f"{'=' * 60}")
    for metric in ["ndcg@1", "ndcg@5", "ndcg@10", "recall@1", "recall@5", "mrr", "map@10"]:
        if metric in agg:
            print(f"  {metric:<16} {agg[metric]['mean']:>10.3f}")
    qlat = results["latency"]["text_embed_ms"]
    print(f"\n  Query latency: {qlat['median']:.0f}ms median")
    if errors:
        print(f"  Errors: {len(errors)}")
    print(f"\n  Saved: {out}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Video Embedding Benchmark")
    parser.add_argument("--model", type=str, help="gemini, marengo, internvideo, nova, siglip")
    parser.add_argument("--all", action="store_true", help="Run all models")
    args = parser.parse_args()

    if not QUERIES_FILE.exists():
        print("Dataset not found. Run download_dataset.py first.")
        sys.exit(1)

    if args.all:
        models = ["gemini", "marengo", "xclip", "nova", "siglip", "internvideo2"]
    elif args.model:
        models = [args.model]
    else:
        parser.print_help()
        sys.exit(1)

    STORES_MODELS = {"mixedbread"}

    all_results = {}
    for name in models:
        try:
            if name in STORES_MODELS:
                r = run_stores_benchmark(name)
            else:
                r = run_benchmark(name)
            if r:
                all_results[name] = r
        except Exception as e:
            print(f"\nFailed: {name}: {e}")
            import traceback
            traceback.print_exc()

    if len(all_results) > 1:
        combined = RESULTS_DIR / "all_models.json"
        with open(combined, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nCombined: {combined}")
        print("Run `python report.py` for the comparison table.")


if __name__ == "__main__":
    main()
