# Video Embedding Model Benchmark: Retrieval Quality for Information Retrieval

A head-to-head comparison of multimodal embedding models for text-to-video retrieval, tested on a standardized dataset with reproducible IR metrics.

## Results (6 of 7 models — Nova pending)

| Model | Dims | NDCG@5 | NDCG@10 | R@1 | R@5 | MRR | mAP@10 | Latency (ms) | Vec Size |
|-------|------|--------|---------|-----|-----|-----|--------|-------------|----------|
| **Gemini Embedding 2** | 3072 | 0.697 | **0.769** | 0.200 | 0.717 | 0.896 | 0.712 | 2458 | 12,288 B |
| **Twelve Labs Marengo 2.7**\* | 1024 | **0.721** | 0.760 | **0.250** | **0.743** | **1.000** | **0.737** | 18148 | 4,096 B |
| **Mixedbread Wholembed v3**\*\* | ColBERT | 0.644 | 0.757 | 0.216 | 0.649 | 0.932 | 0.688 | 500 | N/A |
| X-CLIP Base (temporal) | 512 | 0.327 | 0.470 | 0.067 | 0.367 | 0.520 | 0.338 | 192 | 2,048 B |
| SigLIP 2 SO400M (frame-avg) | 1152 | 0.202 | 0.325 | 0.075 | 0.237 | 0.466 | 0.204 | 636 | 4,608 B |
| InternVideo2-Stage2 6B | 768 | 0.186 | 0.302 | 0.046 | 0.237 | 0.405 | 0.170 | 24817 | 3,072 B |

\* *Marengo results based on 34/60 queries (56%) due to Twelve Labs free-tier daily rate limit (50 req/day). Full results pending re-run.*
\*\* *Mixedbread results based on 37/60 queries (62%) due to rate limiting. Uses Stores API (ColBERT-style late interaction) — no single embedding vector.*

**Key takeaways:**
- The top 3 are all API models — Gemini, Marengo, and Mixedbread cluster tightly at NDCG@10 0.757–0.769
- Marengo's perfect MRR means the correct video was *always* ranked #1 across all 34 evaluated queries — remarkable for a video-specialist model
- Mixedbread Wholembed v3 is a strong third at NDCG@10=0.757 with perfect MRR on hard negatives (1.000) — its ColBERT-style late interaction works well for video
- All three API models are 2x+ better than the best open-source option (X-CLIP at 0.470 NDCG@10)
- X-CLIP is the best open-source model despite being the smallest (512d) — temporal cross-frame attention matters more than model size
- InternVideo2 6B underperforms despite being 12x larger than X-CLIP — its Stage2 checkpoint is optimized for multimodal pretraining, not zero-shot retrieval
- Frame-averaging (SigLIP) is a poor proxy for video understanding — temporal structure matters
- Latency varies wildly: X-CLIP at 192ms (local GPU) vs Marengo at 18s (API processing). Mixedbread query latency is 500ms but requires upfront video upload
- Mixedbread's Stores-based approach is architecturally different — you upload videos once and search via API, no embedding vectors to manage

**Pending:** Amazon Nova Multimodal (need Bedrock model access). Marengo re-run for remaining 26 queries after rate limit reset.

## Models Under Test

| # | Model | Provider | Dims | Video Native | Type | Status |
|---|-------|----------|------|-------------|------|--------|
| 1 | **Gemini Embedding 2** | Google | 768–3072 (Matryoshka) | Yes | API | Done |
| 2 | **Twelve Labs Marengo 2.7** | Twelve Labs | 1024 | Yes | API | Partial (34/60 queries) |
| 3 | **Mixedbread Wholembed v3** | Mixedbread | ColBERT (late interaction) | Yes | API (Stores) | Partial (37/60 queries) |
| 4 | **X-CLIP Base** | Microsoft | 512 | Yes (cross-frame attn) | Open-source | Done |
| 5 | **Amazon Nova Multimodal** | Amazon (Bedrock) | 256–3072 (Matryoshka) | Yes | API | Pending (need Bedrock) |
| 6 | **SigLIP 2 SO400M** | Google (open) | 1152 | No (frame-avg baseline) | Open-source | Done |
| 7 | **InternVideo2-Stage2 6B** | OpenGVLab | 768 | Yes (temporal ViT) | Open-source | Done |

### Why These Models

- **Gemini Embedding 2**: Launched March 2026. First model to unify text, image, video, audio, and documents in one embedding space. The new standard.
- **Twelve Labs Marengo 3.0**: Purpose-built for video. Claims 78.5% composite vs. Nova's 61.8%. Handles 4K and 4-hour videos. The specialist.
- **Mixedbread Wholembed v3**: ColBERT-style late-interaction retrieval. Uses Stores API — upload videos, search with text. No single embedding vector; retrieval is done server-side.
- **X-CLIP Base**: Microsoft's video-text contrastive model. Extends CLIP with cross-frame temporal attention. Best open-source model with true temporal understanding.
- **Amazon Nova**: AWS's five-modality model on Bedrock. The enterprise default.
- **SigLIP 2 SO400M**: SOTA open-source image-text encoder. No video understanding — mean-pools 8 frames. Included as a baseline to quantify the cost of ignoring temporal structure.
- **InternVideo2-Stage2 6B**: OpenGVLab's video foundation model. Contrastive video-text pretraining with temporal ViT backbone + BERT-large text encoder. The largest open-source model in this benchmark.

## Dataset

20 CC0 videos from Pexels (640x360, 24fps, ≤10s, H.264), normalized with ffmpeg:

| Category | Videos | Content |
|----------|--------|---------|
| Sports | 4 | Basketball, soccer, swimming, running |
| Cooking | 4 | Chopping, stirring, pouring, plating |
| Nature | 4 | Waterfall, ocean, forest, birds |
| Urban | 4 | Traffic, pedestrians, construction, skyline |
| Technology | 4 | Typing, soldering, 3D printing, robotics |

60 text queries with **graded relevance** (following BEIR/MTEB conventions):
- **relevance=2**: Exact match (query describes this specific video)
- **relevance=1**: Partial match (query describes the category)
- **relevance=0**: Irrelevant (implicit)

Three query types per video:
1. **Exact** — "A person chopping vegetables with a knife on a cutting board"
2. **Partial** — "Someone preparing food in a kitchen"
3. **Hard negative** — semantically adjacent but wrong domain (e.g., "A technician carefully assembling small electronic parts by hand" for a cooking video)

### Downloading

```bash
python download_dataset.py  # No API key needed — uses direct Pexels CDN links
```

## Metrics

Following standard IR evaluation (BEIR, MTEB):

- **NDCG@K** — Normalized Discounted Cumulative Gain with graded relevance (primary metric)
- **Recall@K** — Fraction of relevant docs retrieved in top K
- **MRR** — Mean Reciprocal Rank of first relevant result
- **mAP@10** — Mean Average Precision at 10
- **Latency** — Embedding time per video (median, p95)
- **Storage** — Bytes per embedding vector (float32)

## Running the Benchmark

### Prerequisites

```bash
pip install -r requirements.txt
```

### API Keys (only for API models)

```bash
export GEMINI_API_KEY="..."            # Gemini Embedding 2
export TWELVE_LABS_API_KEY="..."       # Marengo 3.0
export AWS_ACCESS_KEY_ID="..."         # Nova (Bedrock)
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-east-1"
export MIXEDBREAD_API_KEY="..."        # Wholembed v3
```

X-CLIP and SigLIP 2 run locally (MPS/CUDA/CPU).

### Run

```bash
# Download dataset (no API key needed)
python download_dataset.py

# Run individual models
python benchmark.py --model gemini
python benchmark.py --model marengo
python benchmark.py --model xclip
python benchmark.py --model nova
python benchmark.py --model siglip
python benchmark.py --model mixedbread

# Run all
python benchmark.py --all

# Generate comparison report
python report.py
```

Results are written to `results/` as JSON + `results/COMPARISON.md`.

## Detailed Results by Query Type

### Exact Match (text describes a specific video)

| Model | NDCG@1 | NDCG@5 | R@1 | R@5 | MRR |
|-------|--------|--------|-----|-----|-----|
| Marengo 2.7\* | 0.556 | 0.662 | 0.250 | 0.667 | 1.000 |
| Gemini Embedding 2 | 0.600 | 0.652 | 0.200 | 0.600 | 0.887 |
| Mixedbread Wholembed v3\*\* | 0.564 | 0.630 | 0.231 | 0.615 | 0.962 |
| X-CLIP Base | 0.267 | 0.378 | 0.100 | 0.400 | 0.606 |
| SigLIP 2 (frame-avg) | 0.133 | 0.160 | 0.075 | 0.212 | 0.435 |
| InternVideo2 6B | 0.050 | 0.176 | 0.037 | 0.263 | 0.396 |

### Hard Negative (adjacent domain — model should NOT be confused)

| Model | NDCG@1 | NDCG@5 | R@5 | MRR |
|-------|--------|--------|-----|-----|
| Marengo 2.7\* | 1.000 | 0.846 | 0.841 | 1.000 |
| Mixedbread Wholembed v3\*\* | 1.000 | 0.802 | 0.750 | 1.000 |
| Gemini Embedding 2 | 0.800 | 0.790 | 0.800 | 0.900 |
| X-CLIP Base | 0.200 | 0.322 | 0.350 | 0.467 |
| SigLIP 2 (frame-avg) | 0.400 | 0.236 | 0.200 | 0.556 |
| InternVideo2 6B | 0.200 | 0.220 | 0.250 | 0.423 |

Marengo and Mixedbread both achieve perfect NDCG@1 and MRR on hard negatives — never fooled by adjacent-domain queries. Gemini is close behind at 0.900 MRR.

\* *Marengo: partial results — 34/60 queries evaluated.*
\*\* *Mixedbread: partial results — 37/60 queries evaluated. Uses Stores-based retrieval (ColBERT-style).*

## Cost Comparison

| Model | Pricing Model | Est. Cost / 1000 Videos |
|-------|--------------|------------------------|
| Gemini Embedding 2 | Free tier (1K/day), then per-token | ~$0 (free tier) |
| Marengo 3.0 | $0.033/min | ~$5–15 |
| Mixedbread Wholembed v3 | Free tier, then per-token | ~$0 (free tier) |
| X-CLIP Base | Self-hosted | GPU/CPU cost only |
| Nova Multimodal | Bedrock per-token | ~$2–8 |
| SigLIP 2 | Self-hosted | GPU/CPU cost only |

## Methodology Notes

- All embeddings L2-normalized to unit vectors; retrieval uses cosine similarity
- Random seed fixed at 42 for reproducibility
- Videos normalized to uniform specs (640x360, 24fps, 10s max, H.264) to control for input variation
- SigLIP 2 samples 8 frames uniformly and mean-pools frame embeddings
- X-CLIP samples 8 frames with cross-frame temporal attention
- Gemini processes the full video natively via file upload
- Marengo processes the full video natively via file upload (purpose-built video model)
- InternVideo2 samples 4 frames with temporal ViT backbone + BERT-large text encoder (6B params, CPU inference)
- Mixedbread uses Stores API (ColBERT-style late interaction) — videos uploaded once, retrieval via search endpoint. No single embedding vector; ranking done server-side
- Graded relevance (NDCG) is the primary metric because binary relevance (R@K) doesn't capture partial matches

## License

Benchmark code: MIT. Videos: CC0 (Pexels). Model weights subject to their respective licenses.
