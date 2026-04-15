"""Generate Marengo-vs-Gemini comparison tables by query type.

Reads results/marengo_3.json and results/gemini_embedding_2.json and prints
markdown tables formatted for direct paste into README.md.

Usage:
    python compare.py
"""

import json
from pathlib import Path

RESULTS = Path(__file__).parent / "results"

MODELS = [
    ("marengo_3.json", "Twelve Labs Marengo 3.0"),
    ("gemini_embedding_2.json", "Gemini Embedding 2"),
]

# (section heading, query-type key, [(column title, metric key)])
SECTIONS = [
    (
        "Exact Match (text describes a specific video)",
        "exact",
        [
            ("NDCG@1", "ndcg@1"),
            ("NDCG@5", "ndcg@5"),
            ("R@1", "recall@1"),
            ("R@5", "recall@5"),
            ("MRR", "mrr"),
        ],
    ),
    (
        "Partial Match (category-level query)",
        "partial",
        [
            ("NDCG@1", "ndcg@1"),
            ("NDCG@5", "ndcg@5"),
            ("R@1", "recall@1"),
            ("R@5", "recall@5"),
            ("MRR", "mrr"),
        ],
    ),
    (
        "Hard Negative (adjacent domain — model should NOT be confused)",
        "hard_negative",
        [
            ("NDCG@1", "ndcg@1"),
            ("NDCG@5", "ndcg@5"),
            ("R@5", "recall@5"),
            ("MRR", "mrr"),
        ],
    ),
]


def fmt(val: float | None) -> str:
    return "—" if val is None else f"{val:.3f}"


def main() -> None:
    models = [
        (display_name, json.loads((RESULTS / filename).read_text()))
        for filename, display_name in MODELS
    ]

    for heading, qtype, cols in SECTIONS:
        print(f"### {heading}\n")
        print("| Model | " + " | ".join(title for title, _ in cols) + " |")
        print("|-------|" + "|".join(["--------"] * len(cols)) + "|")
        for display_name, data in models:
            bqt = data.get("by_query_type", {}).get(qtype, {})
            cells = [fmt(bqt.get(metric, {}).get("mean")) for _, metric in cols]
            print(f"| {display_name} | " + " | ".join(cells) + " |")
        print()


if __name__ == "__main__":
    main()
